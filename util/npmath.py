#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains various simple mathmatical tools that can be used for different data analysis.
"""

import numpy as np


def solve_linear_equation(
    expression: str, 
    var: str='x',
    )->float:
    """
    Description
    -----------
    Quickly solve a linear equation using complex number.

    NOTE
    ----
    x + 10 = x - 10 type equation will lead to division by zero error

    Example
    -------
    >> eq = 'x - 3*x - 46 = 10 - x'
    >> -56.0

    """
    _complex = eval((expression.replace("=", "-(") + ")").replace(var, "1j"))
    return -_complex.real/_complex.imag


if __name__ == "__main__":
    # Usage of solve_linear_equation
    expr = 'x - 3*x - 46 = 10 - x'
    print(f"evaluating {expr} results to:\n{solve_linear_equation(expr)}")