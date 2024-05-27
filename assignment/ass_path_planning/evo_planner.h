#ifndef __EVO_PLANNER__
#define __EVO_PLANNER__ 1

#include <bits/stdc++.h>
#include <ctime>
#include <random>
#include <cstdlib>
#ifndef MAX_DIS
#define MAX_DIS INT_MAX/8-1
#endif
#ifndef BLOCK
#define BLOCK 0
#endif

// #define GROUP_SIZE 50
#define MUTATION_RATE 0.25

class evo_planner{
public:
    std::vector<std::vector<int>> graph={
        {BLOCK, 12, 10, MAX_DIS, MAX_DIS, MAX_DIS, 12},
        {12, BLOCK, 8, 12, MAX_DIS, MAX_DIS, MAX_DIS},
        {10, 8, BLOCK, 11, 3, MAX_DIS, 9},
        {MAX_DIS, 12, 11, BLOCK, 11, 10, MAX_DIS},
        {MAX_DIS, MAX_DIS, 3, 11, BLOCK, 6, 7},
        {MAX_DIS, MAX_DIS, MAX_DIS, 10, 6, BLOCK, 9},
        {12, MAX_DIS, 9, MAX_DIS, 7, 9, BLOCK}
    };

    void run(int epochs = 100, int terminal = 20){
        for(int i = 0; i < epochs; ++i){
            _multiply();
            // std::cout << "end multiply\n";
            _quick_sort(this->group_cost, this->groups, 0, this->groups.size()-1);
            // std::cout << "end sort\n";

            // kick the last one out of the group
            this->groups.pop_back();
            this->group_cost.pop_back();
            if(i % terminal == 0){
                // print the whold group, using printf
                printf("epoch %d\n", i);
                for(int i = 0; i < this->groups.size(); ++i){
                    printf("group %d: ", i);
                    for(int j = 0; j < this->groups[i].size(); ++j){
                        printf("%d ", this->groups[i][j]);
                    }
                    printf("cost: %d\n", this->group_cost[i]);
                }
            }
        }
        this->path = this->groups[0];
        this->cost = this->group_cost[0];
    }

    std::vector<int> getPath(){
        return this->path;
    }

    int getCost(){
        return this->cost;
    }

    evo_planner(int GROUP_SIZE = 50){
        this->GROUP_SIZE = GROUP_SIZE;
        this->hasVisited = std::vector<bool>(7);
        this->path = std::vector<int>();
        this->cost = 0;
        this->evo_directions = {
            {1,2,6},
            {0,2,3},
            {0,1,3,4,6},
            {1,2,4,5},
            {2,3,5,6},
            {3,4,6},
            {0,2,4,5}
        };
        this->N = 7;
        u = std::uniform_int_distribution<int>(0, 6);
        e.seed(time(0));
        this->_initGroup();

        // print the initial group
        for(int i = 0; i < this->groups.size(); ++i){
            std::cout << "group " << i << ": ";
            for(int j = 0; j < this->groups[i].size(); ++j){
                std::cout << this->groups[i][j] << " ";
            }
            std::cout << "cost: " << this->group_cost[i] << std::endl;
        }
    }

private:
    int cost, N;
    int GROUP_SIZE;
    
    std::default_random_engine e;
    std::uniform_int_distribution<int> u;

    std::vector<int> path;
    std::vector<bool> hasVisited;
    std::vector<std::vector<int>> evo_directions;
    std::vector<std::vector<int>> groups;
    std::vector<int> group_cost;

    void _shuffle(std::vector<int>&vec){
        for(int i = 0; i < vec.size(); ++i){
            int j = u(e) % vec.size();
            std::swap(vec[i], vec[j]);
        }
    }

    int _calCost(std::vector<int>&path){
        int cost = 0;
        for(int i = 0; i < path.size()-1; ++i){
            cost += this->graph[path[i]][path[i+1]];
        }
        cost += this->graph[path[path.size()-1]][path[0]];
        return cost;
    }

    void _quick_sort(std::vector<int>&group_cost, std::vector<std::vector<int>>&group, int left, int right){
        if(left >= right){
            return;
        }
        int pv = group_cost[left];
        int i = left, j = right;
        while(i < j){
            while(i < j && group_cost[j] >= pv){
                j--;
            }
            // group_cost[i] = group_cost[j];
            std::swap(group_cost[i], group_cost[j]);
            std::swap(group[i], group[j]);
            while(i < j && group_cost[i] < pv){
                i++;
            }
            std::swap(group_cost[i], group_cost[j]);
            std::swap(group[i], group[j]);
        }
        if(i > left){
            _quick_sort(group_cost, group, left, i-1);
        }
        if(j < right){
            _quick_sort(group_cost, group, j+1, right);
        }
    }

    void _multiply(){
        // randomly choose 2 parents
        int father_id = u(e) % GROUP_SIZE;
        int mother_id = u(e) % GROUP_SIZE;
        while(mother_id == father_id){
            mother_id = u(e) % GROUP_SIZE;
        }
        
        std::vector<int> father = this->groups[father_id];
        std::vector<int> mother = this->groups[mother_id];

        int save_len = u(e) % 4;
        if(save_len == 0) save_len = 1;
        if(save_len > 4) save_len = 4;
        // randomly select save_len elements from father
        std::vector<int> save;
        for(int i = 0; i < save_len; ++i){
            int id = u(e) % 7;
            while(std::find(save.begin(), save.end(), id) != save.end()){
                id = u(e) % 7;
            }
            save.push_back(id);
        }
        std::sort(save.begin(), save.end());

        std::vector<int> child(7, -1);

        // set the saved elements to the child
        for(int i = 0; i < save.size(); ++i){
            child[save[i]] = father[save[i]];
        }

        for(int i = 0; i < 7; ++i){
            if(i == save[i]) continue;
            if(std::find(child.begin(), child.end(), mother[i]) == save.end()){
                child[i] = mother[i];
            }
        }

        int mother_start_index = 0;
        for(int i = 0; i < 7; ++i){
            if(child[i] == -1){
                for(int j = mother_start_index; j < 7; ++j){
                    if(std::find(child.begin(), child.end(), mother[j]) == child.end()){
                        child[i] = mother[j];
                        mother_start_index = j+1;
                        break;
                    }
                }
            }
        }

        if(double(u(e)) / u.max() < MUTATION_RATE){
            // printf("mutation !\n");
            _mutate(child);
        }
        
        int child_cost = _calCost(child);
        this->groups.push_back(child);
        this->group_cost.push_back(child_cost);
    }

    void _mutate(std::vector<int>& path){
        int idx1 = u(e) % N;
        int idx2 = u(e) % N;
        while(idx1 == idx2){
            idx2 = u(e) % N;
        }
        std::swap(path[idx1], path[idx2]);
    }

    void _initGroup(){
        this->groups = std::vector<std::vector<int>>();
        this->group_cost = std::vector<int>();
        this->group_cost.reserve(GROUP_SIZE);
        for(int i = 0; i < GROUP_SIZE; ++i){
            std::vector<int> path = _init_single_path();
            this->groups.push_back(path);
            this->group_cost.push_back(_calCost(path));
        }
    }

    std::vector<int> _init_single_path() {
        std::vector<int> path;
        std::vector<bool> visited(graph.size(), false);
        std::default_random_engine e;
        std::uniform_int_distribution<int> u(0, graph.size() - 1);

        // Start from a random node
        int current = u(e);
        path.push_back(current);
        visited[current] = true;

        while (path.size() < graph.size()) {
            std::vector<int> neighbors;

            // Find all unvisited neighbors
            for (int i = 0; i < graph.size(); ++i) {
                if (graph[current][i] != MAX_DIS && !visited[i]) {
                    neighbors.push_back(i);
                }
            }

            // If there are no unvisited neighbors, this is a dead end.
            // This should not happen if the graph is connected.
            if (neighbors.empty()) {
                break;
            }

            // Choose a random neighbor
            std::uniform_int_distribution<int> u(0, neighbors.size() - 1);
            current = neighbors[u(e)];

            path.push_back(current);
            visited[current] = true;
        }

        return path;
    }
};


#endif