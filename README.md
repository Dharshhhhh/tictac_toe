# GPU vs GPU Tic-Tac-Toe (CUDA)

## Project Description
This project implements a Tic-Tac-Toe game where two GPU-based agents compete using CUDA.

- GPU 0 uses a parallel scoring strategy (center > corner > edge)
- GPU 1 uses a random move strategy

The CPU coordinates the game and updates the board.

## Code Description
A CUDA kernel is used to evaluate all possible moves in parallel.
Each thread evaluates one position and assigns a score.
The CPU selects the best move based on these scores.

## Demonstration
The program prints the board after every move and announces the winner.

## How to Run
Compile:
```
nvcc tictactoe.cu -o game
```

Run:
```
./game
```
