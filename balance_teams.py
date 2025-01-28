#!/usr/bin/env python3

import argparse
import logging
import sys
import math
import statistics
from itertools import permutations

# -----------------------------------------------------------------------------
# 1. LOGGING CONFIGURATION (No external install needed)
# -----------------------------------------------------------------------------
YELLOW = "\033[93m"
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Show WARNING in yellow for illustration
        if record.levelno == logging.WARNING:
            record.msg = f"{YELLOW}{record.msg}{RESET}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)

# -----------------------------------------------------------------------------
# 2. ARGUMENT PARSING
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Distribute players into teams with balanced sums of ratings."
    )
    # Positional arguments
    parser.add_argument(
        "file",
        help="Path to input file containing lines of '<Name> <Rating>'."
    )
    parser.add_argument(
        "team_size",
        type=int,
        help="Size of each team."
    )
    # Optional argument for choosing the algorithm
    parser.add_argument(
        "--algorithm",
        choices=["greedy", "brute_force", "branch_and_bound"],
        default=None,
        help=(
            "Algorithm for distributing players. "
            "Options: 'greedy', 'brute_force', 'branch_and_bound'. "
            "Default is 'branch_and_bound'."
        )
    )

    args = parser.parse_args()

    players = read_players_from_file(args.file)
    if not players:
        logger.error("No valid players found. Exiting.")
        sys.exit(1)

    num_players = len(players)
    if num_players < args.team_size:
        logger.warning(
            f"There are only {num_players} players, but team_size is {args.team_size}. "
            "Exiting."
        )
        sys.exit(1)

    if not args.algorithm:
        if num_players > 45:
            args.algorithm = "greedy"
        elif num_players > 10:
            args.algorithm = "branch_and_bound"
        else:
            args.algorithm = "brute_force"
        logger.info(f"using algorithm {args.algorithm}")

    if num_players % args.team_size != 0:
        print(f"WARNING: number of players {num_players} is not divisible by team size "
              f"{args.team_size}")

    return (args, players, num_players)

# -----------------------------------------------------------------------------
# 3. READ PLAYERS (NAME, RATING) FROM FILE
# -----------------------------------------------------------------------------
def read_players_from_file(file_path):
    players = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Skipping malformed line #{line_num}: {line}")
                continue
            name = " ".join(parts[:-1])  # everything except last token
            try:
                rating = float(parts[-1])
            except ValueError:
                logger.warning(f"Skipping line #{line_num} with non-float rating: {line}")
                continue
            players.append((name, rating))
    return players

# -----------------------------------------------------------------------------
# 4. ALGORITHM IMPLEMENTATIONS
# -----------------------------------------------------------------------------

def greedy_balance_teams(players, team_size):
    """
    A simple greedy approach:
    1) Sort players by descending rating.
    2) Always place the next highest-rated player into the team with the lowest current sum (not at capacity).
    """
    num_players = len(players)
    num_teams = math.ceil(num_players / team_size)

    # Sort by descending rating
    sorted_players = sorted(players, key=lambda x: x[1], reverse=True)

    teams = [[] for _ in range(num_teams)]
    team_sums = [0.0] * num_teams

    for (name, rating) in sorted_players:
        # choose the team with the smallest sum that isn't full
        idx_min = None
        min_sum = float('inf')
        for i in range(num_teams):
            if len(teams[i]) < team_size:
                if team_sums[i] < min_sum:
                    min_sum = team_sums[i]
                    idx_min = i
        teams[idx_min].append((name, rating))
        team_sums[idx_min] += rating

    return teams

def brute_force_balance_teams(players, team_size):
    """
    Brute-force approach by checking all permutations.
    Only feasible for small num_players (like <= 10).
    """
    num_players = len(players)
    num_teams = math.ceil(num_players / team_size)

    if num_players == 0:
        return []

    best_teams = None
    best_diff = float('inf')

    for perm in permutations(players, num_players):
        # chunk the permutation into teams of size `team_size`
        perm_teams = []
        idx = 0
        for _ in range(num_teams):
            team = perm[idx:idx+team_size]
            perm_teams.append(team)
            idx += team_size
            if idx >= num_players:
                break
        # compute difference
        team_sums = [sum(x[1] for x in t) for t in perm_teams]
        diff = max(team_sums) - min(team_sums)
        if diff < best_diff:
            best_diff = diff
            best_teams = perm_teams

    logger.info(f"Brute Force: Minimum difference found = {best_diff:.2f}")
    return best_teams

def branch_and_bound_balance_teams(players, team_size):
    """
    A Branch & Bound approach that tries to assign each player to a team.
    We keep track of partial sums and prune if we can't possibly beat
    the current best arrangement.

    WARNING: Still exponential in the worst case, but often faster than
             checking all permutations. Works reasonably for medium-sized
             (<= ~15-20) numbers of players.
    """
    num_players = len(players)
    # Number of teams (last one can be smaller if num_players isn't divisible by team_size)
    num_teams = math.ceil(num_players / team_size)

    # Sort players by descending rating (helps bounding heuristics)
    sorted_players = sorted(players, key=lambda x: x[1], reverse=True)

    # Best result so far
    best_diff = float('inf')
    teams_best = None

    # Prepare data structures:
    # - assignment[i] = which team player i belongs to
    # - team_sums[t] = total rating sum for team t
    # - team_counts[t] = how many players in team t

    def best_case_difference(team_sums, team_counts, next_player_idx):
        """
        Compute a "best-case" (lowest possible) difference if we distribute the
        remaining players from next_player_idx onward in an *ideal* way.
        Heuristic:
          - For each remaining player (in descending rating order),
            assign them to the *currently smallest sum* team that isn't full.
        Then return the difference (max_sum - min_sum).
        If this best-case difference is still >= best_diff, we know we can't improve,
        so we prune.
        """

        sums_copy = team_sums[:]
        counts_copy = team_counts[:]
        # Distribute each remaining player to the smallest sum team that isn't full
        for i in range(next_player_idx, num_players):
            rating = sorted_players[i][1]
            # Find the team with the smallest sum that isn't full
            min_team_idx = None
            min_sum_value = float('inf')
            for t in range(num_teams):
                if counts_copy[t] < team_size:
                    if sums_copy[t] < min_sum_value:
                        min_sum_value = sums_copy[t]
                        min_team_idx = t
            if min_team_idx is not None:
                sums_copy[min_team_idx] += rating
                counts_copy[min_team_idx] += 1

        return max(sums_copy) - min(sums_copy)

    def backtrack(i, assignment, team_sums, team_counts):
        nonlocal teams_best, best_diff

        # If we've assigned all players, evaluate final difference
        if i == num_players:
            current_diff = max(team_sums) - min(team_sums)
            if current_diff < best_diff:
                best_diff = current_diff
                # Reconstruct the final teams
                final_teams = [[] for _ in range(num_teams)]
                for player_idx, team_idx in enumerate(assignment):
                    final_teams[team_idx].append(sorted_players[player_idx])
                teams_best = final_teams
            return

        # Prune using a "best-case difference" estimate
        # If even distributing the remaining players optimally won't beat best_diff, prune.
        possible_best_diff = best_case_difference(team_sums, team_counts, i)
        if possible_best_diff >= best_diff:
            return

        # Try assigning player i to each team (if capacity allows)
        (pname, prating) = sorted_players[i]
        for t in range(num_teams):
            if team_counts[t] < team_size:
                # Choose team t for player i
                old_sum = team_sums[t]
                old_count = team_counts[t]

                assignment[i] = t
                team_sums[t] += prating
                team_counts[t] += 1

                # Recurse for next player
                backtrack(i + 1, assignment, team_sums, team_counts)

                # Undo assignment (backtrack)
                team_sums[t] = old_sum
                team_counts[t] = old_count
                assignment[i] = -1

    assignment = [-1] * num_players
    team_sums_init = [0.0] * num_teams
    team_counts_init = [0] * num_teams

    backtrack(0, assignment, team_sums_init, team_counts_init)

    logger.info(f"Branch & Bound: Minimum difference found = {best_diff:.2f}")
    return teams_best

def analyze_team_ratings(teams):
    # Calculate total ratings for each team
    total_ratings = [sum(member[1] for member in team) for team in teams]
    
    # Compute max difference and standard deviation
    max_diff = max(total_ratings) - min(total_ratings)
    stddev = statistics.stdev(total_ratings)
    
    return max_diff, stddev

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    args, players, num_players = parse_args()

    # Choose algorithm
    if args.algorithm == "greedy":
        teams = greedy_balance_teams(players, args.team_size)
        maxdiff, stddev = analyze_team_ratings(teams)
        logger.info(f"'greedy' algorithm found teams with max diff = {maxdiff:.2f} and "
            f"standard deviation = {stddev:.2f}")
    elif args.algorithm == "branch_and_bound":
        # This is still exponential, but prunes a lot of permutations
        if num_players > 45:
            logger.warning("Branch & Bound may be very slow for n > 45.")
        teams1 = branch_and_bound_balance_teams(players, args.team_size)

        # check just in case if the greedy is better than branch_and_bound, shouldn't be
        teams2 = greedy_balance_teams(players, args.team_size)
        maxdiff1, stddev1 = analyze_team_ratings(teams1)
        maxdiff2, stddev2 = analyze_team_ratings(teams2)
        logger.info(f"'branch_and_bound' algorithm found teams with max diff = {maxdiff1:.2f} and "
                     f"standard deviation = {stddev1:.2f}")
        if maxdiff1 < maxdiff2:
            teams = teams1
        else:
            teams = teams2
            logger.info(f"'greedy' algorithm found teams with max diff = {maxdiff2:.2f} and "
                        f"standard deviation = {stddev2:.2f}")
            logger.info("auto-selecting 'greedy' teams")
    elif args.algorithm == "brute_force":
        if num_players > 10:
            logger.warning("Brute force may be extremely slow for n > 10.")
        teams = brute_force_balance_teams(players, args.team_size)
    else:
        logger.error(f"Unknown algorithm {args.algorithm}")
        sys.exit(1)

    # Print final teams
    if teams is None:
        logger.error("No teams were formed. Exiting.")
        sys.exit(1)

    # Show results
    print()
    for idx, team in enumerate(teams, start=1):
        total_rating = sum(member[1] for member in team)
        names = [member[0] for member in team]
        print(f"Team {idx:>2}: {', '.join(names):<30} Total Rating = {total_rating:.2f}")


if __name__ == "__main__":
    main()
