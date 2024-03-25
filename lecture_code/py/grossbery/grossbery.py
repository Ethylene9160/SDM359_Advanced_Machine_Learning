import sympy as sp
import random

def generate_polynomials(num_variables, num_equations, max_degree):
    variables = sp.symbols('x0:%d' % num_variables)
    polynomials = []
    for _ in range(num_equations):
        degree = random.randint(1, max_degree)
        terms = [random.randint(-10, 10) * sp.product(random.choice(variables)**random.randint(1, 3), (var, 1, random.randint(1, degree))) for var in variables]
        polynomial = sum(terms)
        polynomials.append(polynomial)
    return variables, polynomials


def compute_groebner_basis(polynomials, variables):
    return sp.groebner(polynomials, *variables, order='lex')

def generate_data(num_variables, num_train_equations, num_test_equations, max_degree):
    train_variables, train_polynomials = generate_polynomials(num_variables, num_train_equations, max_degree)
    test_variables, test_polynomials = generate_polynomials(num_variables, num_test_equations, max_degree)
    
    train_basis = compute_groebner_basis(train_polynomials, train_variables)
    test_basis = compute_groebner_basis(test_polynomials, test_variables)
    
    return train_variables, train_polynomials, train_basis, test_variables, test_polynomials, test_basis

if __name__ == '__main__':
    num_variables = 2
    num_train_equations = 5
    num_test_equations = 2
    max_degree = 3

    train_variables, train_polynomials, train_basis, test_variables, test_polynomials, test_basis = generate_data(num_variables, num_train_equations, num_test_equations, max_degree)

    print("Train Variables:", train_variables)
    print("Train Polynomials:", train_polynomials)
    print("Train Groebner Basis:", train_basis)
    print()
    print("Test Variables:", test_variables)
    print("Test Polynomials:", test_polynomials)
    print("Test Groebner Basis:", test_basis)
