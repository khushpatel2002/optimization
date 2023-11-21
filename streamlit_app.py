import streamlit as st
from solution import vogel, russell, calculate_total_cost, north_west_corner, Problem


def parse_input(input_str, expected_length):
    """Parse input string into a list of integers."""
    try:
        data = list(map(int, input_str.split()))
        if len(data) != expected_length:
            raise ValueError
        return data
    except ValueError:
        st.error(f"Invalid input. Please enter {expected_length} integers.")
        return None


def format_result(method_name, approximation, total_cost):
    result_str = f"{method_name}:\n"
    result_str += f"Plan: {approximation.plan}\n"
    result_str += f"Total Transportation Cost: {total_cost}\n"
    return result_str


def main():
    st.title(" Optimization Assignment 3")
    st.markdown("### Transportation Problem Solver")

    sample_sources = 3
    sample_destinations = 4
    sample_costs = [[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20]]
    sample_supply = [7, 9, 18]
    sample_demand = [5, 8, 7, 14]

    num_sources = st.number_input(
        "Number of Sources", min_value=1, value=sample_sources
    )
    num_destinations = st.number_input(
        "Number of Destinations", min_value=1, value=sample_destinations
    )

    costs = []
    for i in range(num_sources):
        default_cost = (
            " ".join(map(str, sample_costs[i]))
            if i < len(sample_costs)
            else "0 " * num_destinations
        )
        cost_row = st.text_input(
            f"Costs from Source {i + 1} (separate with spaces)", value=default_cost
        )
        parsed_row = parse_input(cost_row, num_destinations)
        if parsed_row is not None:
            costs.append(parsed_row)

    default_supply = (
        " ".join(map(str, sample_supply))
        if num_sources == len(sample_supply)
        else "0 " * num_sources
    )
    supply_input = st.text_input(
        "Supply for each source (separate with spaces)", value=default_supply
    )
    supply = parse_input(supply_input, num_sources)

    default_demand = (
        " ".join(map(str, sample_demand))
        if num_destinations == len(sample_demand)
        else "0 " * num_destinations
    )
    demand_input = st.text_input(
        "Demand for each destination (separate with spaces)", value=default_demand
    )
    demand = parse_input(demand_input, num_destinations)

    if st.button("Solve Transportation Problem"):
        if None not in costs and supply is not None and demand is not None:
            try:
                problem = Problem(num_sources, num_destinations, costs, supply, demand)
                st.text(problem.format_transportation_table())
                st.markdown("---")

                # North-West Corner Method
                nw_result = north_west_corner(problem)
                total_cost_nw = calculate_total_cost(nw_result.plan, problem.costs)
                nw_output = format_result(
                    "North-West Corner Method", nw_result, total_cost_nw
                )
                st.text(nw_output)
                st.text(nw_result.format_result_table("North-West Corner Method"))
                st.markdown("---")

                # Vogel's Approximation Method
                vogel_result = vogel(problem)
                total_cost_vogel = calculate_total_cost(
                    vogel_result.plan, problem.costs
                )
                vogel_output = format_result(
                    "Vogel's Approximation Method", vogel_result, total_cost_vogel
                )
                st.text(vogel_output)
                st.text(
                    vogel_result.format_result_table("Vogel's Approximation Method")
                )

                st.markdown("---")
                # Russell's Approximation Method
                russell_result = russell(problem)
                total_cost_russell = calculate_total_cost(
                    russell_result.plan, problem.costs
                )

                russell_output = format_result(
                    "Russell's Approximation Method", russell_result, total_cost_russell
                )
                st.text(russell_output)
                st.text(
                    russell_result.format_result_table("Russell's Approximation Method")
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
