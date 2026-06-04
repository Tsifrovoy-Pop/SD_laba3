#include <iostream>
#include <vector>
using namespace std;

class MyStack {
private:
    vector<long long> data;

public:
    void push(long long x) {
        data.push_back(x);
    }

    void pop() {
        if (!data.empty())
            data.pop_back();
    }

    long long top() const {
        return data.back();
    }

    bool empty() const {
        return data.empty();
    }
};

int main() {
    setlocale(LC_ALL, "Russian");
    system("chcp 65001 > nul");

    cout << "Введите количество элементов: \n";
    int N;
    cin >> N;

    vector<long long> A(N);
    cout << "Введите элементы: \n";
    for (int i = 0; i < N; ++i) cin >> A[i];

    vector<long long> ans(N);
    MyStack st;

    for (int i = N - 1; i >= 0; --i) {
        while (!st.empty() && st.top() <= A[i]) {
            st.pop();
        }
        ans[i] = st.empty() ? 0 : st.top();
        st.push(A[i]);
    }

    cout << "Ответ:\n";
    for (int i = 0; i < N; ++i)
        cout << ans[i] << ' ';
}
