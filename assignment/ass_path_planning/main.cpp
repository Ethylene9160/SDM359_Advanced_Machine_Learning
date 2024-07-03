#include<bits/stdc++.h>
// #include"planner.h"
#include"enhanced_evo_planner.h"
using std::cin;
using std::cout;
#include "uwb_data.h"
std::unordered_map<int, std::string> id2name = {
    {0, "A"},
    {1, "B"},
    {2, "C"},
    {3, "D"},
    {4, "E"},
    {5, "F"},
    {6, "G"}
};

int main(){
    //uwb_data data(1, 2, 3);
    uint8_t d[] = {0,0,0,0,0,0,8,0,1,2,3,4,5,6,7,8,0,0,0};
    printf("size of d:%d\n", sizeof(d));
    uwb_data data(d);
    //std::vector<uint8_t> d = {1, 2, 3, 4, 5};
    //data.set_data(d);
    auto data2 = data.get_data();
    for(auto i : data2){
        cout << (int)i << " ";
    }
    serials s = data.serialize();
    cout << "\n";
    for(int i = 0; i < s.len; ++i){
        cout << (int)s.data_ptr[i] << " ";
    }
    return 0;
}

int main233(int a, char* args[]){
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