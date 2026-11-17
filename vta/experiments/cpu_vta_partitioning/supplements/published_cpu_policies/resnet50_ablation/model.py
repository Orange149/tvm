"""Scalar oracle retains all fifteen subsets and the existing accounting."""
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score as original_score
from .common import settings


def costs_for(key, records, programs=None, ddr=None):
    return Costs(records, programs=programs, ddr=ddr, distribution=settings(key)['distribution'])


def score(configuration, costs, key):
    return original_score(configuration, costs, settings(key)['scalar_key'])
