import time
from itertools import combinations
from typing import List

import networkx as nx
import numpy as np

from .metadata import Text2ImageMetaData, Text2ThreeDMetaData, Text2VideoMetaData, Text2VisionMetaData
from .scene_graph import get_sg_desc
from .utils import make_and_description


def get_element_num_dict(graph):
	object_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "object_node"
	]
	attribute_nodes = [
		(n, d) for n, d in graph.nodes(data=True) if d["type"] == "attribute_node"
	]
	relation_edges = [
		(n1, n2, d)
		for n1, n2, d in graph.edges(data=True)
		if d.get("type") == "relation_edge"
	]
	return {
		"object"   : len(object_nodes),
		"attribute": len(attribute_nodes),
		"relation" : len(relation_edges),
	}


def convert_sg_to_json(graph: nx.DiGraph):
	nodes = list(graph.nodes(data=True))
	edges = list(graph.edges(data=True))
	graph = {
		"nodes": nodes,
		"edges": edges,
	}
	return graph


def convert_json_to_sg(graph_json: dict):
	graph = nx.DiGraph()
	graph.add_nodes_from(graph_json["nodes"])
	graph.add_edges_from(graph_json["edges"])
	return graph


def find_isomorphisms(graph1, graph2):
	# find whether a scene_graph is subgraph of the other scene_graph or not.
	def node_match(n1, n2):
		return n1["type"] == n2["type"]

	matching_subgraphs = []
	for sub_nodes in combinations(graph2.nodes(), len(graph1.nodes())):
		subG = graph2.subgraph(sub_nodes)

		GM = nx.algorithms.isomorphism.DiGraphMatcher(
			subG, graph1, node_match=node_match
		)
		if GM.is_isomorphic():
			matching = {v: k for k, v in GM.mapping.items()}
			matching_subgraphs.append(matching)

	return matching_subgraphs


def add_seed_graph_to_template_graph(
		seed_graph: nx.DiGraph, template_graph: nx.DiGraph
):
	if seed_graph is not None:
		conditioned_templates = []
		match_subgraphs = find_isomorphisms(seed_graph, template_graph)
		for match_subgraph in match_subgraphs:
			scene_graph = template_graph.copy()
			for seed_node, template_node in match_subgraph.items():
				scene_graph.nodes[template_node]["value"] = seed_graph.nodes[seed_node]["value"]
				for seed_neighbor in seed_graph[seed_node]:
					if (
							seed_graph.nodes[seed_neighbor]["type"] == "object_node"
							and seed_neighbor in match_subgraph
					):
						template_neighbor = match_subgraph[seed_neighbor]
						if scene_graph.has_edge(template_node, template_neighbor):
							scene_graph.edges[template_node, template_neighbor]["value"] \
								= seed_graph.edges[seed_node, seed_neighbor]["value"]
			conditioned_templates.append(scene_graph)
		return conditioned_templates
	else:
		return [template_graph]


def get_global_attribute_desc(global_attributes):
	global_attributes = [
		f"{v}"
		for _, v in global_attributes.items()
	]
	return make_and_description(global_attributes)


def get_prompt(global_attribute_desc, sg_desc, generate_type):
	if global_attribute_desc == "":
		return f"Create a {generate_type}: {sg_desc}"
	else:
		return f"Create a {global_attribute_desc} {generate_type}: {sg_desc}"


class Text2VisionPromptGenerator():
	generate_type = "vision"
	metadata: Text2VisionMetaData

	def __init__(self, metadata: Text2VisionMetaData, seed=42):
		self.metadata = metadata
		self.seed = seed
		self.rng = np.random.default_rng(seed=seed)

	def _task_plan_to_str(self, task_plan):
		return get_sg_desc(task_plan["scene_graph"])

	def _complete_sg(self, scene_graph: nx.DiGraph):
		assert isinstance(scene_graph, nx.DiGraph)
		for node, data in scene_graph.nodes(data=True):
			if "value" not in data:
				if data["type"] == "attribute_node":
					k, v = self.metadata.sample_metadata(
						self.rng, element_type="attribute"
					)
					data["value_type"] = k
					data["value"] = v
				elif data["type"] == "object_node":
					data["value"] = self.metadata.sample_metadata(
						self.rng, element_type="object"
					)
		for u, v, data in scene_graph.edges(data=True):
			if "value" not in data:
				if data.get("type") == "relation_edge":
					k, v = self.metadata.sample_metadata(
						self.rng, element_type="relation"
					)
					data["value_type"] = k
					data["value"] = v
		return scene_graph

	def _sample_scene_graph(self, complexity, seed_graph, seed_graph_element_num_dict, element_num_dict, retry=50):
		sg_templates = self.metadata.query_sg_templates(
			complexity, seed_graph_element_num_dict, element_num_dict
		)
		if len(sg_templates) == 0:
			raise ValueError("No specific template scene graph found")

		conditioned_template = None
		for i in self.rng.permutation(len(sg_templates)):
			template_graph = sg_templates[i]
			conditioned_templates = add_seed_graph_to_template_graph(
				seed_graph, template_graph
			)
			# randomly pick one of the conditioned templates
			if len(conditioned_templates) != 0:
				index = self.rng.integers(len(conditioned_templates))
				conditioned_template = conditioned_templates[index]
				break

		if conditioned_template is None:
			raise ValueError("No template scene graph matches seed graph")

		scene_graph = self._complete_sg(conditioned_template)
		return scene_graph

	def _sample_global_attributes(self, number_of_global_attributes):
		return self.metadata.sample_global_attribute(self.rng, number_of_global_attributes)

	def sample_task_plans(
			self,
			complexity=5,
			number_of_global_attributes=1,
			sample_numbers=100,
			time_limit=60,
			seed_graph: nx.DiGraph = None,
			element_num_dict: dict = None,
	) -> List:

		# check whether user input is legal

		if seed_graph is None:
			seed_graph = nx.DiGraph()
		seed_graph_element_num_dict = get_element_num_dict(seed_graph)
		assert sum(seed_graph_element_num_dict.values()) <= complexity

		if element_num_dict is None:
			element_num_dict = {
				"object"   : None,
				"attribute": None,
				"relation" : None,
			}
		n_elements = 0
		for k in ['object', 'relation', 'attribute']:
			if element_num_dict[k] is not None:
				assert seed_graph_element_num_dict[k] <= element_num_dict[k]
				n_elements += element_num_dict[k]
		assert n_elements <= complexity

		# sample

		task_plans = []
		start_time = time.time()
		while len(task_plans) < sample_numbers:
			# make sure the time limit is not exceeded
			if time.time() - start_time > time_limit:
				print("Time limit: 60s exceeded. Exiting the sampling process.")
				break
			scene_graph = self._sample_scene_graph(complexity, seed_graph, seed_graph_element_num_dict, element_num_dict)
			global_attributes = self._sample_global_attributes(number_of_global_attributes)
			scene_graph_str = convert_sg_to_json(scene_graph)
			task_plans.append(
				{
					"global_attributes": global_attributes,
					"scene_graph"      : scene_graph_str,
				}
			)
		print(f"sampling {len(task_plans)} task plans.")
		return task_plans

	def _generate_task(self, task_plan):
		scene_graph = convert_json_to_sg(task_plan["scene_graph"])
		sg_desc = get_sg_desc(scene_graph)
		global_attribute_desc = get_global_attribute_desc(task_plan["global_attributes"])
		prompt = get_prompt(global_attribute_desc, sg_desc, self.generate_type)
		return prompt

	def generate(self, task_plan, seed=None):
		if seed is not None:
			self.rng = np.random.default_rng(seed=seed)
		prompt = self._generate_task(task_plan)

		task = {
			"prompt"           : prompt,
			"global_attributes": task_plan["global_attributes"],
			"scene_graph"      : task_plan["scene_graph"],
		}
		return task


class Text2ImagePromptGenerator(Text2VisionPromptGenerator):
	generate_type = "image"

	def __init__(self, metadata: Text2ImageMetaData, seed=42):
		super().__init__(metadata, seed=seed)


class Text2VideoPromptGenerator(Text2VisionPromptGenerator):
	generate_type = "video"

	def __init__(self, metadata: Text2VideoMetaData, seed=42):
		super().__init__(metadata, seed=seed)


class Text2ThreeDScenePromptGenerator(Text2VisionPromptGenerator):
	generate_type = "3D scene"

	def __init__(self, metadata: Text2ThreeDMetaData, seed=42):
		super().__init__(metadata, seed=seed)


class Text2ThreeDObjectPromptGenerator(Text2VisionPromptGenerator):
	generate_type = "3D object"

	def __init__(self, metadata: Text2ThreeDMetaData, seed=42):
		super().__init__(metadata, seed=seed)

	def sample_task_plans(
			self,
			complexity=5,
			number_of_global_attributes=1,
			sample_numbers=100,
			time_limit=60,
			object_graph: nx.DiGraph = None,  # a graph which have 1 object node and its attribute nodes.
			element_num_dict=None,
	):
		# for 3D objects, there is not relation in the scene graph and there is only 1 objects
		return super().sample_task_plans(
			complexity=complexity,
			number_of_global_attributes=number_of_global_attributes,
			sample_numbers=sample_numbers,
			time_limit=time_limit,
			seed_graph=object_graph,
			element_num_dict={"object": 1, "attribute": None, "relation": 0},
		)
