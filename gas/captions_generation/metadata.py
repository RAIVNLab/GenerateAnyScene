import json
import operator
import os
import pickle
from itertools import combinations, permutations

import networkx as nx
import pandas as pd


# this function takes in a term in metadata and normalized it by removing all the '_' '|' in the metadata
def normalized_term(term):
	return term.replace("_", " ").replace("|", " ")

def sort_scene_attributes(scene_attributes):
	right_order = [
		"weather", "location", "genre", "artist", "painting style", "technique",
		"lighting", "size", "view", "depth of focus", "camera model", "camera gear",
		"camera movement", "focal length", "ISO", "aperture", "video editing style",
		"time span", "3D shape attribute"
	]
	
	# Create a list to store keys in the desired order
	sorted_keys = []
	
	# Add keys that are in right_order and exist in scene_attributes
	for key in right_order:
		if key in scene_attributes:
			sorted_keys.append(key)
	
	# Add keys that are not in right_order
	for key in scene_attributes:
		if key not in sorted_keys:
			sorted_keys.append(key)
	
	# Create a new dictionary with keys in sorted order
	sorted_scene_attributes = {key: scene_attributes[key] for key in sorted_keys}
	
	return sorted_scene_attributes

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
		# load basic data
		self.attributes = json.load(
			open(os.path.join(path_to_metadata, "attributes.json"))
		)
		self.objects = json.load(
			open(os.path.join(path_to_metadata, "objects.json"))
		)
		self.relations = json.load(
			open(os.path.join(path_to_metadata, "relations.json"))
		)
		self.scene_attributes = json.load(
			open(os.path.join(path_to_metadata, "scene_attributes.json"))
		)
		# set sg_template_path
		self.path_to_sg_template = path_to_sg_template
		self.sg_template_store_dict = {}

	def get_object_type(self, object_value):
		if object_value not in self.objects_category:
			raise ValueError(f"Object value {object_value} not found in metadata")
		else:
			return self.objects_category[object_value]

	def get_available_elements(self, element_type):
		if element_type == "object":
			return self.objects
		elif element_type == "attribute":
			return self.attributes
		elif element_type == "relation":
			return self.relations


	# Implement allowed_topic later
	def sample_scene_attribute(self, rng, n, allowed_scene_attributes):

		scene_attributes = {}
		available_scene_attributes = []
		
		for attr_type in self.scene_attributes:
			if attr_type in allowed_scene_attributes:
				# recompute, first compute the 
				if allowed_scene_attributes[attr_type] == "all":
					# allow all the subtyle in this attr tyle
					for sub_type in self.scene_attributes[attr_type]:
						available_scene_attributes.append((attr_type, sub_type))
				else:
					# only allow the subtype in the allowed_scene_attributes
					for allowed_sub_type in allowed_scene_attributes[attr_type]:
						available_scene_attributes.append((attr_type, allowed_sub_type))
	  	
		assert n <= len(available_scene_attributes), f"n should be less than the number of scene attributes: {len(available_scene_attributes)}"
		

		scene_attribute_selections = rng.choice(available_scene_attributes, n, replace=False)
		for scene_attribute_selection in scene_attribute_selections:
			scene_attribute_type = str(scene_attribute_selection[0])
			scene_attribute_sub_type = str(scene_attribute_selection[1])
			attributes = self.scene_attributes[scene_attribute_type][scene_attribute_sub_type]
			# TODO: take the intersection of allow_attribute and allowed scene attributes
			scene_attributes[scene_attribute_sub_type] = str(rng.choice(attributes))
		return sort_scene_attributes(scene_attributes)

	def sample_metadata(self, rng, element_type):
		if element_type == "object":
			return str(rng.choice(list(self.objects)))

		elif element_type == "attribute":
			available_attributes = self.attributes
			available_attributes_list = []
			for type, value_list in available_attributes.items():
				for value in value_list:
					available_attributes_list.append(f"{type}|{value}")
			attr_type, attr_value = rng.choice(available_attributes_list).split("|")
			return attr_type, attr_value

		elif element_type == "relation":
			available_relations = self.relations
			available_relations_list = []
			for type, value_list in available_relations.items():
				for value in value_list:
					available_relations_list.append(f"{type}|{value}")
			rel_type, rel_val = rng.choice(available_relations_list).split("|")
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


class Text2VideoMetaData(Text2VisionMetaData):
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		super().__init__(path_to_metadata, path_to_sg_template)

class Text2ThreeDMetaData(Text2VisionMetaData):
	def __init__(self, path_to_metadata, path_to_sg_template=None):
		super().__init__(path_to_metadata, path_to_sg_template)
