#ifndef UI_HPP
#define UI_HPP

#include "Reversi.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <cctype>
using namespace std;

class UI
{
public:
    UI() = default;
    // 运行游戏
    void Launch();
    // 人类玩家是否先手
    bool GetPriority();
    // 获得人类玩家的决策
    int GetInput(const Reversi &game);
    // 输出棋盘
    void PrintBoard(const Reversi &game);
};

#endif