#include<bits/stdc++.h>
// #include"planner.h"
#include"enhanced_evo_planner.h"
using std::cout;
using std::cin;

int main(int a, char* args[]){
    evo_planner p;
    int cost = 0;
    // p.findMinCircle(0, cost);
    // p.dpmethod();
    p.run(500000, 100000);
    auto output = p.getPath();
    cost = p.getCost();

    cout<<"results showing: \n";
    for(int i = 0; i < output.size(); ++i){
        cout << output[i] << " ";
    }
    cout << "\ncost: " << cost << std::endl;

    return 0;
}