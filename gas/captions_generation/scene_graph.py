from collections import defaultdict

import inflect
import networkx as nx

from .utils import make_and_description


def label_repeated_objects_in_sg(graph: nx.DiGraph):
	# this function is to find the same objects in the scene_graph, like there are 2 "apple" in the sg.
	# then in the caption we can refer to them as "the first apple" and "the second apple" so it won't be confusing.
	object_nodes = [
		n for n, d in graph.nodes(data=True) if d.get("type") == "object_node"
	]
	grouped_nodes = defaultdict(list)
	for node in object_nodes:
		value = graph.nodes[node].get("value")
		attributes = get_attributes(graph, node)
		key = (value, tuple(attributes))
		grouped_nodes[key].append(node)

	same_nodes_groups = {
		key: nodes for key, nodes in grouped_nodes.items() if len(nodes) > 1
	}
	for nodes in same_nodes_groups.values():
		for index, node in enumerate(nodes):
			graph.nodes[node]["is_repeated"] = index
	for node in object_nodes:
		if "is_repeated" not in graph.nodes[node]:
			graph.nodes[node]["is_repeated"] = "no"

	return same_nodes_groups


def get_attributes(graph, node):
	assert graph.nodes[node]["type"] == "object_node"
	attributes = []
	for neighbor in graph.neighbors(node):
		if graph.nodes[neighbor].get("type") == "attribute_node":
			attributes.append(graph.nodes[neighbor].get("value"))
	attributes.sort()
	return attributes


def topsort(graph: nx.DiGraph):
	# notice that this topsort only care about "object_node" in the graph
	object_nodes = [
		n for n, d in graph.nodes(data=True) if d.get("type") == "object_node"
	]
	subgraph = graph.subgraph(object_nodes).copy()
	try:
		topo_order = list(nx.topological_sort(subgraph))
		return topo_order
	except nx.NetworkXUnfeasible:
		print("The subgraph contains a cycle and cannot be topologically sorted.")
		return []


def mention_node(graph, node):
	if "mentioned" not in graph.nodes[node]:
		graph.nodes[node]["mentioned"] = True


def get_attr_obj_desc(graph, node) -> str:
	inflect_engine = inflect.engine()
	name = graph.nodes[node]["value"]

	object_desc = ""
	if graph.nodes[node]["is_repeated"] != "no":
		object_desc += inflect_engine.ordinal(graph.nodes[node]["is_repeated"] + 1) + " "

	attributes_desc = make_and_description(get_attributes(graph, node))
	if attributes_desc != "":
		object_desc += attributes_desc + " "

	object_desc += name

	if "mentioned" not in graph.nodes[node] and graph.nodes[node]["is_repeated"] == "no":
		if not inflect_engine.singular_noun(name):
			object_desc = inflect_engine.a(object_desc)
	else:
		object_desc = "the" + " " + object_desc

	return object_desc


def get_relation_desc(graph, node) -> str:
	relations_desc = []
	relation_to_targets = defaultdict(list)
	for head, target, data in graph.out_edges([node], data=True):
		if data["type"] == "relation_edge":
			relation_to_targets[data["value"]].append(target)

	for relation, targets in relation_to_targets.items():
		# add mentioned flag to both nodes
		mention_node(graph, node)
		for target in targets:
			mention_node(graph, target)

		targets = [get_attr_obj_desc(graph, target) for target in targets]
		target_desc = make_and_description(targets)
		relations_desc.append(
			f"is {relation} {target_desc}"
		)

	return make_and_description(relations_desc)


def get_sg_desc(scene_graph):
	label_repeated_objects_in_sg(scene_graph)
	topsort_order = topsort(scene_graph)
	templates = []
	for node in topsort_order:
		attr_obj = get_attr_obj_desc(scene_graph, node)
		relations_desc = get_relation_desc(scene_graph, node)
		if relations_desc != "":
			templates.append(attr_obj + " " + relations_desc)
		else:
			if 'mentioned' not in scene_graph.nodes[node]:
				templates.append("there is " + attr_obj)

	return "; ".join(templates) + '.'