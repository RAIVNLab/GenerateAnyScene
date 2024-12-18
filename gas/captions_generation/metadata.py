import json
import operator
import os
import pickle
from itertools import combinations, permutations

import networkx as nx
import pandas as pd



def has_cycle(graph):
	try:
		nx.find_cycle(graph, orientation="original")
		return True
	except:
		return False


def combinations_with_replacement_counts(n, r):
	size = n + r - 1
	for indices in combinations(range(size), n - 1):
		starts = [0] + [index + 1 for index in indices]
		stops = indices + (size,)
		yield tuple(map(operator.sub, stops, starts))


def _enumerate_template_graphs(complexity, graph_store):
	cnt = 0
	for obj_num in range(1, complexity + 1):

		graph = nx.DiGraph()
		# Add nodes for each object
		for obj_id in range(1, obj_num + 1):
			graph.add_node(f"object_{obj_id}", type="object_node")

		possible_relations = list(permutations(range(1, obj_num + 1), 2))
		for rel_num in range(min(complexity - obj_num, len(possible_relations)) + 1):
			attr_num = complexity - obj_num - rel_num
			obj_attr_combo = combinations_with_replacement_counts(obj_num, attr_num)

			if rel_num == 0:
				for obj_attrs in obj_attr_combo:
					g = graph.copy()
					for obj_id, obj_attr_num in enumerate(obj_attrs):
						for attr_id in range(1, obj_attr_num + 1):
							g.add_node(
								f"attribute|{obj_id + 1}|{attr_id}",
								type="attribute_node",
							)
							g.add_edge(
								f"object_{obj_id + 1}",
								f"attribute|{obj_id + 1}|{attr_id}",
								type="attribute_edge",
							)
					graph_store.add_digraph(
						{
							"object"   : obj_num,
							"attribute": attr_num,
							"relation" : rel_num,
						},
						g,
					)
					cnt += 1
			else:

				rel_combo = combinations(possible_relations, rel_num)

				for rels in rel_combo:

					rel_graph = graph.copy()

					for obj_id1, obj_id2 in rels:
						rel_graph.add_edge(
							f"object_{obj_id1}",
							f"object_{obj_id2}",
							type="relation_edge",
						)

					if has_cycle(rel_graph):
						continue

					for obj_attrs in obj_attr_combo:
						g = rel_graph.copy()
						for obj_id, obj_attr_num in enumerate(obj_attrs):
							for attr_id in range(1, obj_attr_num + 1):
								g.add_node(
									f"attribute|{obj_id + 1}|{attr_id}",
									type="attribute_node",
								)
								g.add_edge(
									f"object_{obj_id + 1}",
									f"attribute|{obj_id + 1}|{attr_id}",
									type="attribute_edge",
								)
						graph_store.add_digraph(
							{
								"object"   : obj_num,
								"attribute": attr_num,
								"relation" : rel_num,
							},
							g,
						)
						cnt += 1

	print(
		f"finished enumerate scene graph templates, total number of templates: {cnt}"
	)


class SGTemplateStore:
	def __init__(self, complexity):
		self.graph_store = []
		self.df = pd.DataFrame(
			columns=[
				"idx",
				"numbers_of_objects",
				"numbers_of_attributes",
				"numbers_of_relations",
			]
		)
		self.complexity = complexity

	def __len__(self):
		return len(self.graph_store)

	def add_digraph(self, element_num_dict, digraph):
		# idx start from zero, so idx = len(self.graph_store)
		idx = len(self.graph_store)
		self.graph_store.append(digraph)
		new_row = pd.DataFrame({
			'idx'                  : [idx],
			'numbers_of_objects'   : [element_num_dict['object']],
			'numbers_of_attributes': [element_num_dict['attribute']],
			'numbers_of_relations' : [element_num_dict['relation']]
		})
		self.df = pd.concat([self.df, new_row], ignore_index=True)

	def query_digraph(self, seed_graph_element_num_dict, element_num_dict):
		conditions = []
		for k in ['object', 'relation', 'attribute']:
			if k in element_num_dict and element_num_dict[k] is not None:
				conditions.append(f'numbers_of_{k}s == {element_num_dict[k]}')
			else:
				conditions.append(f'numbers_of_{k}s >= {seed_graph_element_num_dict[k]}')

		query = " and ".join(conditions)

		if query:
			queried_df = self.df.query(query)
		else:
			queried_df = self.df

		indices_of_query_graph = queried_df["idx"].tolist()
		result_graphs = [self.graph_store[idx] for idx in indices_of_query_graph]
		return result_graphs

	def save(self, path_to_store):
		assert len(self.graph_store) == len(self.df)
		pickle.dump(self.graph_store, open(os.path.join(path_to_store, f"template_graph_complexity{self.complexity}.pkl"), "wb"))
		pickle.dump(self.df, open(os.path.join(path_to_store, f"template_graph_features_complexity{self.complexity}.pkl"), "wb"))

	def load(self, path_to_store):
		if os.path.exists(os.path.join(path_to_store, f"template_graph_complexity{self.complexity}.pkl")) and os.path.exists(os.path.join(path_to_store, f"template_graph_features_complexity{self.complexity}.pkl")):
			self.graph_store = pickle.load(open(os.path.join(path_to_store, f"template_graph_complexity{self.complexity}.pkl"), "rb"))
			self.df = pickle.load(open(os.path.join(path_to_store, f"template_graph_features_complexity{self.complexity}.pkl"), "rb"))
			if len(self.graph_store) == len(self.df):
				print("Loading sg templates from cache successfully")
				return True

		print("Loading failed, re-enumerate sg templates")
		return False


class Text2VisionMetaData():
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		self.attributes = json.load(
			open(os.path.join(path_to_metadata, "attributes.json"))
		)
		self.objects = json.load(
			open(os.path.join(path_to_metadata, "objects_list.json"))
		)
		self.relations = json.load(
			open(os.path.join(path_to_metadata, "relations.json"))
		)
		self.path_to_sg_template = path_to_sg_template
		self.sg_template_store_dict = {}

	def sample_global_attribute(self, rng, n=1):
		assert n <= len(self.global_attributes), "n should be less than the number of global attributes"
		global_attributes = {}
		global_attribute_types = rng.choice(list(self.global_attributes.keys()), n, replace=False)
		for global_attribute_type in global_attribute_types:
			attributes = self.global_attributes[global_attribute_type]
			if isinstance(attributes, list):
				global_attributes[str(global_attribute_type)] = str(rng.choice(attributes))
			elif isinstance(attributes, dict):
				global_attribute_sub_type = rng.choice(list(attributes.keys()))
				global_attributes[str(global_attribute_type)] = str(rng.choice(attributes[global_attribute_sub_type]))
			else:
				raise ValueError("Invalid global attribute type")
		return global_attributes

	def sample_metadata(self, rng, element_type):
		if element_type == "object":
			return str(rng.choice(list(self.objects)))
		elif element_type == "attribute":
			attr_type = rng.choice(list(self.attributes.keys()))
			attr_value = str(rng.choice(self.attributes[attr_type]))
			return attr_type, attr_value
		elif element_type == "relation":
			rel_type = rng.choice(list(self.relations.keys()))
			rel_val = str(rng.choice(self.relations[rel_type]))
			return rel_type, rel_val
		else:
			raise ValueError("Invalid type")

	def query_sg_templates(self, complexity, seed_graph_element_num_dict, element_num_dict):
		if self.path_to_sg_template is None:
			# set the default cache path
			if not os.path.exists("./sg_template_cache"):
				os.makedirs("./sg_template_cache")
			self.path_to_sg_template = "./sg_template_cache"

		if complexity not in self.sg_template_store_dict:
			# initialize the store
			self.sg_template_store_dict[complexity] = SGTemplateStore(complexity)
			if not self.sg_template_store_dict[complexity].load(self.path_to_sg_template):
				# if loading the cache failed, re-enumerate the sg templates
				_enumerate_template_graphs(complexity, self.sg_template_store_dict[complexity])
				self.sg_template_store_dict[complexity].save(self.path_to_sg_template)

		sg_templates = self.sg_template_store_dict[complexity].query_digraph(seed_graph_element_num_dict, element_num_dict)
		return sg_templates


class Text2ImageMetaData(Text2VisionMetaData):
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		super().__init__(path_to_metadata, path_to_sg_template)
		self.global_attributes = json.load(
			open(os.path.join(path_to_metadata, "image_attributes.json"))
		)


class Text2VideoMetaData(Text2VisionMetaData):
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		super().__init__(path_to_metadata, path_to_sg_template)
		self.global_attributes = {}
		# video can both use global attributes from image and video
		self.global_attributes['video_unique'] = json.load(
			open(os.path.join(path_to_metadata, "video_attributes.json"))
		)
		self.global_attributes['image_and_video'] = json.load(
			open(os.path.join(path_to_metadata, "image_attributes.json"))
		)
	def sample_global_attribute(self, rng, n=1):
		is_video_unique_attribute = str(rng.choice(list(self.global_attributes.keys())))
		global_attributes = {}
		global_attribute_types = rng.choice(list(self.global_attributes[is_video_unique_attribute].keys()), n, replace=False)
		for global_attribute_type in global_attribute_types:
			attributes = self.global_attributes[is_video_unique_attribute][global_attribute_type]
			if isinstance(attributes, list):
				global_attributes[str(global_attribute_type)] = str(rng.choice(attributes))
			elif isinstance(attributes, dict):
				global_attribute_sub_type = rng.choice(list(attributes.keys()))
				global_attributes[str(global_attribute_type)] = str(rng.choice(attributes[global_attribute_sub_type]))
			else:
				raise ValueError("Invalid global attribute type")
		return global_attributes


class Text2ThreeDMetaData(Text2VisionMetaData):
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		super().__init__(path_to_metadata, path_to_sg_template)
		self.global_attributes = json.load(
			open(os.path.join(path_to_metadata, "3D_attributes.json"))
		)
