# Team Balancer

A Python tool to distribute players into balanced teams based on their ratings using various algorithms. The goal is to minimize the difference in total ratings between teams.

## Features
- Supports **greedy**, **branch-and-bound**, and **brute-force** algorithms.
- Automatically selects the most suitable algorithm based on the number of players.
- Outputs teams and their total ratings with minimal rating differences.

## Usage
```bash
./balance_teams.py <file> <team_size> [--algorithm <algorithm>]
```

or with Windows (you need to install Python from python.org):
```
python balance_teams.py <file> <team_size> [--algorithm <algorithm>]
```

### Arguments:
- `<file>`: Path to the input file containing player data in the format `<Name> <Rating>`.
- `<team_size>`: Number of players per team (integer).
- `--algorithm`: (Optional) Specifies the algorithm to use for team balancing. Options:
  - `greedy`: Fast and simple, but less optimal.
  - `branch_and_bound`: Strikes a balance between speed and accuracy (default for medium-sized datasets).
  - `brute_force`: Exhaustive search, best for very small datasets (≤10 players).  
- `-h, --help`: Displays usage information and exits.

### Example
```bash
./balance_teams.py test_data 3 --algorithm branch_and_bound
```

### Input Format

The input file should have one player per line in the format:

`<Name>` `<Rating>`
