#include<bits/stdc++.h>
// #include"planner.h"
#include"enhanced_evo_planner.h"
using std::cin;
using std::cout;
std::unordered_map<int, std::string> id2name = {
    {0, "A"},
    {1, "B"},
    {2, "C"},
    {3, "D"},
    {4, "E"},
    {5, "F"},
    {6, "G"}
};

int main(int a, char* args[]){
    evo_planner p(30);
    int cost = 0;
    // p.findMinCircle(0, cost);
    // p.dpmethod();
    p.run(500000, 100000);
    auto output = p.getPath();
    cost = p.getCost();

    cout<<"results showing: \n";
    for(int i = 0; i < output.size(); ++i){
        cout << id2name[output[i]] << " -> ";
    }
    cout << id2name[output[0]];
    cout << "\ncost: " << cost << std::endl;
    cin.get();
    return 0;
}