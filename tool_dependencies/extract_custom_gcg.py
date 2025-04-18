#!/usr/bin/env python3

import r2pipe

from graphviz import Source
import tempfile

import networkx as nx
import matplotlib.pyplot as plt

from collections import deque, defaultdict

MAX_NUM_NODES = 500

def save_graph_networkx(G, save_filepath):
    nx.write_graphml(G, save_filepath[:-4] + ".xml")

def save_graph_pdf(dot_graph, save_filepath):
    s = Source(dot_graph)
    s.render(save_filepath[:-4], format='pdf', cleanup=True)

def plot_graph_pdf(dot_graph):
    
    s = Source(dot_graph)
    s.view(tempfile.mktemp('.gv'))

def plot_graph_plt(G):

    options = {
        'node_size': 0,
        'width': 1,
        'arrowsize': 20,
        'font_size': 12,
        'bbox': dict(facecolor = "skyblue")
    }
    plt.figure(1,figsize=(60,15))
    pos = nx.nx_agraph.graphviz_layout(G, prog = "dot", args='-Grankdir=LR')
    nx.draw(G, pos=pos, with_labels=True, **options)
    plt.show()

# This function is used to prepare the features of a node of the graph given the json object of the function returned by radare2
def prepare_features(function):

    features = {}
    features["size"] = function["size"] # size of the function
    features["is-pure"] = int(function["is-pure"] == "true") # is the function pure (has any side effect)?
    features["realsz"] = function["realsz"] # real size of the function (include any padding)
    features["cc"] = function["cc"] # number of calling convention
    features["nbbs"] = function["nbbs"] # number of basic blocks
    features["ninstrs"] = function["ninstrs"] # number of instructions
    features["edges"] = function["edges"] # number of edges
    features["indegree"] = function["indegree"] # number of incoming edges
    features["outdegree"] = function["outdegree"] # number of outgoing edges
    if "nlocals" in function.keys(): # number of local variables
        features["nlocals"] = function["nlocals"]
    else:
        features["nlocals"] = 0
    
    if "nargs" in function.keys(): # number of arguments
        features["nargs"] = function["nargs"]
    else:
        features["nargs"] = 0
    
    # type of the function (extracted from the name assigned by radare2)
    features["type_fcn"] = features["type_int"] = features["type_sym"] = features["type_loc"] = features["type_imp"] = features["type_sub"] = features["type_entry"] = 0
    if function["name"].startswith("entry"):
        features["type_entry"] = 1
    elif function["name"].startswith("fcn"):
        features["type_fcn"] = 1
    elif function["name"].startswith("int"):
        features["type_int"] = 1
    elif function["name"].startswith("sym"):
        features["type_sym"] = 1
    elif function["name"].startswith("loc"):
        features["type_loc"] = 1
    elif function["name"].startswith("sub"):
        features["type_sub"] = 1
    
    if function["name"].startswith("imp") or function["name"][4:].startswith("imp"):
        features["type_imp"] = 1

    return features

# This function extracts the global callgraph manually by iterating on all functions radare2 detects ("aflj" command)
def extract_gcg(filepath, discard = True):

    # Initialize p2pipe instance
    r2 = r2pipe.open(filepath)
    # Set the file format to PE
    r2.cmd("e asm.arch=x86")
    r2.cmd("e asm.os=windows")
    r2.cmd("e anal.in=io.maps")
    r2.cmd("e anal.esil=false")
    r2.cmd("e asm.emu=false")
    # Analyze the stripped binary
    r2.cmd("aaa 2>/dev/null")

    # Get the list of functions of the binary checked by radare2
    functions = r2.cmdj("aflj")
    # Initialize the graph
    G = nx.DiGraph()

    # Add in the graph all entries as nodes
    entries_fcn = [(function["name"], prepare_features(function)) for function in functions if function["name"].startswith("entry")]
    G.add_nodes_from(entries_fcn)

    for function in functions:

        # if more than MAX_NUM_NODES nodes return None (discard the sample)
        if G.number_of_nodes() > MAX_NUM_NODES and discard:
            return None

        # Stop if number of edges is greater than 0 (there is one connected component)
        if G.number_of_edges() > 0:
            break

        to_explore = deque()
        to_explore.append(function)
        explored = defaultdict(lambda: False)

        G.add_node(function["name"], **prepare_features(function))
        
        while len(to_explore) > 0:

            # if more than MAX_NUM_NODES nodes return None (discard the sample)
            if G.number_of_nodes() > MAX_NUM_NODES and discard:
                return None
            
            # Pop from left of the dequeue a function and mark the function as explored
            fcn = to_explore.popleft()

            if explored[fcn["name"]]:
                continue

            explored[fcn["name"]] = True

            # Forward edges
            if "callrefs" in fcn.keys():

                for f in fcn["callrefs"]:
                    cf_list = r2.cmdj(f"afij @ {f['addr']}")
                    if cf_list == []:
                        continue
                    called_function = cf_list[0]
                    if called_function["name"] not in G.nodes():
                        G.add_node(called_function["name"], **prepare_features(called_function))
                    if fcn["name"] not in G.nodes():
                        G.add_node(fcn["name"], **prepare_features(fcn))

                    # Add edge between fcn and the called function
                    if not G.has_edge(fcn["name"], called_function["name"]) and fcn["name"] != called_function["name"]:
                        G.add_edge(fcn["name"], called_function["name"])
                    
                    if not explored[called_function["name"]]:
                        to_explore.append(called_function)

            # Backward edges
            if "codexrefs" in fcn.keys():

                for f in fcn["codexrefs"]:
                    cf_list = r2.cmdj(f"afij @ {f['addr']}")
                    if cf_list == []:
                        continue
                    caller_function = cf_list[0]
                    if caller_function["name"] not in G.nodes():
                        G.add_node(caller_function["name"], **prepare_features(caller_function))
                    if fcn["name"] not in G.nodes():
                        G.add_node(fcn["name"], **prepare_features(fcn))

                    # Add edge between fcn and the called function
                    if not G.has_edge(caller_function["name"], fcn["name"]) and fcn["name"] != caller_function["name"]:
                        G.add_edge(caller_function["name"], fcn["name"])
                    
                    if not explored[caller_function["name"]]:
                        to_explore.append(caller_function)

    # Drop all isolates except for entry and symbols
    to_drop = list(nx.isolates(G))
    to_drop = [node for node in to_drop if "DLL" not in node.upper() and not node.startswith("entry")]
    G.remove_nodes_from(to_drop)

    if G.number_of_nodes() == 0:
        return None

    return G

# This function extracts the global callgraph (with plots) manually by iterating on all functions radare2 detects ("aflj" command)
def extract_gcg_with_plot(filepath):

    G = extract_gcg(filepath)

    dot_format = """
    digraph code {
    rankdir=LR;
    outputorder=edgesfirst;
    graph [bgcolor=white fontname="Courier" splines="curved"];
    node [penwidth=4 fillcolor=white style=filled fontname="Courier Bold" fontsize=14 shape=box];
    edge [arrowhead="normal" style=bold weight=2];"""

    # if sample has been discarded return None
    if G is None:
        return None, dot_format
    
    dot_graph = str(nx.nx_agraph.to_agraph(G)).replace('digraph "" {', dot_format)

    return G, dot_graph