#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_O 2

__global__ void evaluateMoves(int *board, int *scores)
{
    int i = threadIdx.x;

    if (i >= 9) return;

    if (board[i] == EMPTY)
    {
        if (i == 4) scores[i] = 5;
        else if (i == 0 || i == 2 || i == 6 || i == 8)
            scores[i] = 3;
        else
            scores[i] = 1;
    }
    else
    {
        scores[i] = -1;
    }
}

void printBoard(int *b)
{
    for (int i = 0; i < 9; i++)
    {
        if (b[i] == EMPTY) printf("- ");
        else if (b[i] == PLAYER_X) printf("X ");
        else printf("O ");

        if (i % 3 == 2) printf("\n");
    }
    printf("\n");
}

int checkWinner(int *b)
{
    int winPatterns[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };

    for(int i = 0; i < 8; i++)
    {
        int a = winPatterns[i][0];
        int c = winPatterns[i][1];
        int d = winPatterns[i][2];

        if (b[a] != EMPTY && b[a] == b[c] && b[c] == b[d])
            return b[a];
    }

    return 0;
}

int isBoardFull(int *b)
{
    for(int i = 0; i < 9; i++)
        if(b[i] == EMPTY) return 0;
    return 1;
}

int main()
{
    int board[9] = {0};
    int *d_board, *d_scores;
    int scores[9];

    cudaMalloc((void**)&d_board, 9 * sizeof(int));
    cudaMalloc((void**)&d_scores, 9 * sizeof(int));

    srand(time(NULL));

    int turn = PLAYER_X;
    int winner = 0;

    printf("GPU vs GPU Tic-Tac-Toe\n\n");

    while (!winner && !isBoardFull(board))
    {
        if (turn == PLAYER_X)
        {
            printf("GPU 0 (X) Turn:\n");

            cudaMemcpy(d_board, board, 9 * sizeof(int), cudaMemcpyHostToDevice);

            evaluateMoves<<<1, 9>>>(d_board, d_scores);

            cudaMemcpy(scores, d_scores, 9 * sizeof(int), cudaMemcpyDeviceToHost);

            int bestMove = -1, maxScore = -1;

            for (int i = 0; i < 9; i++)
            {
                if (scores[i] > maxScore)
                {
                    maxScore = scores[i];
                    bestMove = i;
                }
            }

            board[bestMove] = PLAYER_X;
        }
        else
        {
            printf("GPU 1 (O) Turn:\n");

            int move;
            do {
                move = rand() % 9;
            } while (board[move] != EMPTY);

            board[move] = PLAYER_O;
        }

        printBoard(board);

        winner = checkWinner(board);

        turn = (turn == PLAYER_X) ? PLAYER_O : PLAYER_X;
    }

    if (winner == PLAYER_X)
        printf("GPU 0 (X) Wins!\n");
    else if (winner == PLAYER_O)
        printf("GPU 1 (O) Wins!\n");
    else
        printf("It's a Draw!\n");

    cudaFree(d_board);
    cudaFree(d_scores);

    return 0;
}