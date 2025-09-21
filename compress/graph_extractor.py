import torch
import torch.fx as fx

from compress.heuristics import collect_convolution_layers_to_prune
from model.resnet import resnet50


def get_initial_prefix_submodule(graph_module, end_node):
    """
    Extracts a prefix subgraph from the start of the FX GraphModule up to (but not including) `end_node`.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced FX GraphModule from which to extract the prefix subgraph.
    end_node : str
        The name of the node where the prefix subgraph should stop (exclusive).

    Returns
    -------
    prefix_gm : torch.fx.GraphModule
        A new GraphModule containing only the nodes from the start up to `end_node`.
    value_remap : dict
        A mapping from original nodes in `graph_module` to the corresponding nodes
        in the prefix subgraph. Useful for connecting this prefix to subsequent subgraphs.

    Notes
    -----
    - The resulting GraphModule can be called like a normal module, taking the input tensor
      that corresponds to the placeholder node.
    - The output of the prefix subgraph is the last node before `end_node`.
    """
    assert isinstance(graph_module, fx.GraphModule)
    graph = graph_module.graph
    prefix_nodes = []
    prefix_graph = fx.Graph()
    value_remap = {}

    for node in graph.nodes:
        if node.name == end_node:
            break
        else:
            prefix_nodes.append(node)

    assert len(prefix_nodes) > 0, "Prefix nodes must not be empty"

    for node in prefix_nodes:
        new_node = prefix_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    last_node = prefix_nodes[-1]
    prefix_graph.output(value_remap[last_node])

    prefix_gm = fx.GraphModule(root=graph_module, graph=prefix_graph)
    return prefix_gm, value_remap


def get_fx_submodule(graph_module, value_remap, start_node, end_node):
    """
    Extracts a middle subgraph from an FX GraphModule between `start_node` and `end_node`.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced FX GraphModule containing the nodes.
    value_remap : dict
        A mapping from previously copied nodes (e.g., from a prefix subgraph) to the
        corresponding new nodes. Must include any nodes that are inputs to this subgraph.
    start_node : str
        The name of the node where the subgraph should start (inclusive).
    end_node : str
        The name of the node where the subgraph should end (exclusive).

    Returns
    -------
    new_gm : torch.fx.GraphModule
        A new GraphModule containing the nodes between `start_node` and `end_node`.
        Automatically adds placeholder nodes for inputs if needed.
    value_remap : dict
        Updated mapping of original nodes to the corresponding nodes in the new subgraph.
        Can be used to chain multiple subgraph extractions together.

    Notes
    -----
    - Placeholder nodes are automatically created for any input nodes that are not in `value_remap`.
    - The output of the subgraph is set to the last node before `end_node`.
    - The resulting GraphModule can be called with the input tensors corresponding to the placeholders.
    """
    assert isinstance(graph_module, fx.GraphModule)
    assert isinstance(value_remap, dict)
    assert (
        len(value_remap) > 0
    ), "Remap dict cant be empty for slices in the middle of the model"
    graph = graph_module.graph
    new_nodes = []
    new_graph = fx.Graph()
    keep = False

    for node in graph.nodes:
        if node.name == start_node:
            keep = True
        if node.name == end_node:
            break
        if keep:
            new_nodes.append(node)

    assert len(new_nodes) > 0, "Node list must not be empty"

    # Adds placeholder to the beginning of subgraph so that its forward can take an input
    first_node = new_nodes[0]
    for arg in first_node.args:
        if isinstance(arg, fx.Node):
            ph = new_graph.placeholder(f"input_{arg.name}")
            value_remap[arg] = ph

    for node in new_nodes:
        new_node = new_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    last_node = new_nodes[-1]
    new_graph.output(value_remap[last_node])

    new_gm = fx.GraphModule(root=graph_module, graph=new_graph)

    return new_gm, value_remap


def get_suffix_submodule(
    graph_module: fx.GraphModule, value_remap: dict, start_node: str
):
    """
    Extracts the subgraph from `start_node` (inclusive) to the final output of the model.

    Parameters
    ----------
    graph_module : fx.GraphModule
        The FX-traced full model.
    value_remap : dict
        Mapping from previous nodes to their placeholders/substitutes (for start_node input).
    start_node : str
        Name of the node where the suffix begins.

    Returns
    -------
    suffix_gm : fx.GraphModule
        FX GraphModule for the suffix.
    value_remap : dict
        Updated mapping including suffix nodes.
    """
    graph = graph_module.graph
    new_graph = fx.Graph()
    new_nodes = []
    keep = False

    for node in graph.nodes:
        if node.name == start_node:
            keep = True
        if keep:
            new_nodes.append(node)

    assert len(new_nodes) > 0, "Suffix nodes cannot be empty"

    # Add a placeholder for the start node input if it hasn't been mapped yet
    first_node = new_nodes[0]
    for arg in first_node.args:
        if isinstance(arg, fx.Node):
            ph = new_graph.placeholder(f"input_{arg.name}")
            value_remap[arg] = ph

    for node in new_nodes:
        new_node = new_graph.node_copy(node, lambda n: value_remap[n])
        value_remap[node] = new_node

    # Explicitly define output of the subgraph
    last_node = new_nodes[-1]
    new_graph.output(value_remap[last_node])

    suffix_gm = fx.GraphModule(graph_module, new_graph)
    return suffix_gm, value_remap


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    input = torch.randn(1, 3, 224, 224)

    prune_conv_modules, prune_modules_name = collect_convolution_layers_to_prune(
        model=model
    )

    reformatted = []
    for name in prune_modules_name:
        reformatted_name = "_".join(name.split("."))
        reformatted.append(reformatted_name)

    end = reformatted[0]
    gm = fx.symbolic_trace(model)
    prefix_gm, remap = get_initial_prefix_submodule(graph_module=gm, end_node=end)

    start_node = reformatted[0]
    end_node = reformatted[1]

    subgraph, remap_dict = get_fx_submodule(
        graph_module=gm, value_remap=remap, start_node=start_node, end_node=end_node
    )

    out = prefix_gm(input)
    out = subgraph(out)

    print(out.shape)

    outputs = {}

    def hook_fn(model, input, output):
        outputs["maxpool"] = output

    model.maxpool.register_forward_hook(hook_fn)
    model(input)
    print(outputs["maxpool"].shape)

    suffix_module, remap_final = get_suffix_submodule(
        graph_module=gm, value_remap=remap, start_node=end_node
    )
    out = suffix_module(out)
    # print(suffix_module)

    direct_out = model(input)

    assert torch.allclose(out, direct_out)
    print(out.shape)
    print(direct_out.shape)
