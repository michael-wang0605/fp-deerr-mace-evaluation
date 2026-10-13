"""Disable higher derivative graph construction for CHGNet inference only.

Energy, first derivatives, stress, magnetic moments and graph construction are
unchanged. Training/higher derivatives must continue using the original model.
The installed package and public benchmark sources are never edited.
"""
import ast
import hashlib
import inspect
import textwrap
import types


def enable(model):
    assert not model.training, 'Inference optimization must not be used for training'
    original = type(model)._compute
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    replaced = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'grad'
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'autograd'):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'create_graph':
                assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                keyword.value = ast.Constant(value=False)
                replaced += 1
    assert replaced == 2, 'Unexpected CHGNet source; optimization not enabled'
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<validated CHGNet inference>', 'exec'), namespace)
    model._compute = types.MethodType(namespace['_compute'], model)
    return {'optimization': 'inference_create_graph_false', 'changed_keywords': replaced,
            'upstream_compute_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'precision_changed': False, 'retain_graph_changed': False}


def disable(model):
    model._compute = types.MethodType(type(model)._compute, model)
