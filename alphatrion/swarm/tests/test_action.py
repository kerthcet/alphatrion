from typing import Any

import pytest
import yaml

from alphatrion.swarm import ActionNode, GraphStatus, NodeResult, Text, build_graph


def draft(state) -> Text:
    return Text(content="draft")


def shout(state, text: Text, suffix: str) -> Text:
    return Text(content=text.content.upper() + suffix)


def too_long(state, text: Text, limit: int) -> Text | NodeResult:
    if len(text.content) > limit:
        return NodeResult(route="skip", details=f"longer than {limit}")
    return text


FUNCTIONS = {fn.__name__: fn for fn in (draft, shout, too_long)}


def build(pipeline: str):
    return build_graph(yaml.safe_load(pipeline), functions=FUNCTIONS)


@pytest.mark.asyncio
async def test_a_function_with_its_node_s_params():
    result = await build(
        """
        nodes:
          - {name: draft, type: action, fn: draft}
          - {name: shout, type: action, fn: shout, params: {suffix: "!"}}
        """
    ).run()
    assert result.status == GraphStatus.COMPLETED, result.error
    assert result.output == Text(content="DRAFT!")


@pytest.mark.asyncio
async def test_a_function_chooses_the_route():
    result = await build(
        """
        nodes:
          - {name: draft, type: action, fn: draft}
          - name: check
            type: action
            fn: too_long
            params: {limit: 3}
            exit: {skip: end}
        """
    ).run()
    assert result.status == GraphStatus.COMPLETED, result.error
    entry = result.state.memory.history(node="check")[-1]
    assert (entry.data["route"], entry.data["details"]) == ("skip", "longer than 3")


def test_formats():
    node = ActionNode(name="f", fn=too_long, params={"limit": 3})
    build_graph({"nodes": [node]})
    assert node.input_type() is Text
    # too_long's NodeResult isn't a format: it only chooses a route.
    assert node.output_type() is Text
    # A route the function chose outputs what it returned with it.
    assert node.output_type("skip") is Any


def test_params_must_be_given_and_taken():
    with pytest.raises(
        ValueError, match="function shout takes 'suffix', which the node"
    ):
        build("nodes: [{name: f, type: action, fn: shout}]")
    with pytest.raises(ValueError, match="none of its functions takes: sufix"):
        build(
            'nodes: [{name: f, type: action, fn: shout, params: {suffix: "!", sufix: "?"}}]'
        )
