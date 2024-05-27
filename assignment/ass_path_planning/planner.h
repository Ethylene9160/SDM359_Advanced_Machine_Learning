#ifndef _MY_PLANNER_
#define _MY_PLANNER_ 1
#include<bits/stdc++.h>
#define MAX_DIS INT_MAX/2-1
#define BLOCK 0
class Planner{
public:
std::vector<std::vector<int>> map = {
        {BLOCK, 12, 10, MAX_DIS, MAX_DIS, MAX_DIS, 12},
        {12, BLOCK, 8, 12, MAX_DIS, MAX_DIS, MAX_DIS},
        {10, 8, BLOCK, 11, 3, MAX_DIS, 9},
        {MAX_DIS, 12, 11, BLOCK, 11, 10, MAX_DIS},
        {MAX_DIS, MAX_DIS, 3, 11, BLOCK, 6, 7},
        {MAX_DIS, MAX_DIS, MAX_DIS, 10, 6, BLOCK, 9},
        {12, MAX_DIS, 9, MAX_DIS, 7, 9, BLOCK}
    };

    Planner(){
        this->hasVisited = new std::vector<bool>(7);
        this->path = new std::vector<int>();
        this->cost = 0;

        this->N = 7;
        this->M = 1<<(this->N-1);
    }

    ~Planner(){
        delete this->hasVisited;
        delete this->path;
    }

    std::vector<int> findMinCircle(int point, int&cost){
        this->hasVisited = new std::vector<bool>(7);
        this->path = new std::vector<int>();
        this->cost = 0;

        // copy the original graph.
        std::vector<std::vector<int>> graph = this->map;

        // set the start point as visited.
        (*this->hasVisited)[point] = 1;

        this->path->push_back(point);
        for(int i = 0; i < 7; ++i){
            int dis = MAX_DIS;
            int next = -1;
            for(int j = 0; j < 7; ++j){
                if((*this->hasVisited)[j]){
                    continue;
                }
                if(dis > graph[i][j]){
                    dis = graph[i][j];
                    next = j;
                }
            }
            if(next == -1){
                break;
            }
            (*this->hasVisited)[next] = 1;
            this->path->push_back(next);
            this->cost += dis;
        }

        this->cost += graph[this->path->back()][point];
        this->path->push_back(point);
        cost = this->cost;
        return *this->path;
    }

    void dpmethod(){
        this->path = new std::vector<int>();
        this->hasVisited = new std::vector<bool>(7);
        TSP();
        calPath();
    }

    std::vector<int> getPath(){
        return *this->path;
    }

    int getCost(){
        return this->cost;
    }



private:
    // Whether each of the cities has been visited.
    std::vector<bool> *hasVisited;

    // store the path id.
    std::vector<int> *path;

    // cost
    int cost;
    int N;
    int M;


    bool isVisited(){
        for(int i = 1 ; i<N ;i++){
            if((*hasVisited)[i] == false){
                return false;
            }
        }
        return true;
    }


    void calPath(){
        std::vector<std::vector<int>> g = this->map;
        //标记访问数组
        // bool visited[N] = {false};
        int dp[N][M];
        //前驱节点编号
        int pioneer = 0 ,min = MAX_DIS, S = M - 1,temp ;
        //把起点结点编号加入容器
        path->push_back(0);
    
        while(!isVisited()){
            for(int i=1; i<N;i++){
                if((*hasVisited)[i] == false && (S&(1<<(i-1))) != 0){
                    if(min > g[i][pioneer] + dp[i][(S^(1<<(i-1)))]){
                        min = g[i][pioneer] + dp[i][(S^(1<<(i-1)))] ;
                        temp = i;
                    }
                }
            }
            pioneer = temp;
            path->push_back(pioneer);
            (*hasVisited)[pioneer] = true;
            S = S ^ (1<<(pioneer - 1));
            min = MAX_DIS;
        }
    }

    void TSP(){
        std::vector<std::vector<int>> g = this->map;
        int dp[N][M];
        //初始化dp[i][0]
        for(int i = 0 ; i < N ;i++){
            dp[i][0] = g[i][0];
        }
        //求解dp[i][j],先跟新列在更新行
        for(int j = 1 ; j < M ;j++){
            for(int i = 0 ; i < N ;i++ ){
                dp[i][j] = MAX_DIS;
                //如果集和j(或状态j)中包含结点i,则不符合条件退出
                if( ((j >> (i-1)) & 1) == 1){
                    continue;
                }
                for(int k = 1 ; k < N ; k++){
                    if( ((j >> (k-1)) & 1) == 0){
                        continue;
                    }
                    if( dp[i][j] > g[i][k] + dp[k][j^(1<<(k-1))]){
                        dp[i][j] = g[i][k] + dp[k][j^(1<<(k-1))];
                    }
                }
            }
        }
        this->cost = dp[0][M-1];
    }
 
};

#endif