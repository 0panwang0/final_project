#include "UI.hpp"

void UI::Launch()
{
    Reversi game;
    AI bot;

    game.Initialize();
    bool man_round = GetPriority();
    while (!game.IsOver())
    {
        PrintBoard(game);
        if (game.GetAvailable())
        {
            int idx;
            if (man_round)
                idx = GetInput(game);
            else
            {
                idx = bot.Search(game);
                int x=GetX(idx),y=GetY(idx);
                cout << "Bot:" << (char)(x+'0') << (char)(y+'0') << endl;
            }
            game.PlacePiece(idx);
            cout << "Opponent Evaluation:" << game.Evaluate() << endl;
        }
        else {
            game.Skip();
            printf("Skipped!\n");
            char shit[12];
            scanf("%s", shit);
        }
        man_round = man_round ? false : true;
    }
}

bool UI::GetPriority()
{
    string input;
    do
    {
        cout << "your color(b or w):";
        cin >> input;
    } while (input != "b" && input != "w");

    return input == "b";
}

int UI::GetInput(const Reversi &game)
{
    string input;
    while (true)
    {
        cout << ">> ";
        cin >> input;
        if (input.size() != 2 ||
            input[0] < '0' || input[0] > '7' ||
            input[1] < '0' || input[1] > '7')
            continue;

        int x = input[0] - '0', y = input[1] - '0';
        vector<int> next_moves = ULL2Vec(game.GetAvailable());
        int idx1 = GetIndex(x, y);
        int idx2 = lower_bound(next_moves.begin(), next_moves.end(), idx1) - next_moves.begin();
        if (next_moves[idx2] != idx1)
            continue;

        return idx1;
    }
}

void UI::PrintBoard(const Reversi &game)
{
    vector<int> black = ULL2Vec(game.GetBoard(true));
    vector<int> white = ULL2Vec(game.GetBoard(false));
    vector<int> next_moves = ULL2Vec(game.GetAvailable());
    int idx1 = 0, idx2 = 0, idx3 = 0;

    cout << " ";
    for (int i = 0; i < BOARD_WIDTH; i++)
        cout << "   " << (char)(i + '0');
    cout << endl;
    for (int i = 0; i < BOARD_WIDTH; i++)
    {
        cout << "  " << (char)(i + '0') << " ";
        for (int j = 0; j < BOARD_WIDTH; j++)
        {
            int idx = GetIndex(i, j);
            if (idx1 < black.size() && idx == black[idx1])
            {
                cout << "B";
                idx1++;
            }
            else if (idx2 < white.size() && idx == white[idx2])
            {
                cout << "W";
                idx2++;
            }
            else if (idx3 < next_moves.size() && idx == next_moves[idx3])
            {
                //cout << "X";
                cout << " ";
                idx3++;
            }
            else
            {
                cout << " ";
            }

            if (j < BOARD_WIDTH - 1)
                cout << " - ";
            else
                cout << endl;
        }
        if (i < BOARD_WIDTH - 1)
        {
            cout << " ";
            for (int j = 0; j < BOARD_WIDTH; j++)
                cout << "   |";
            cout << endl;
        }
    }
    cout << endl;
}
