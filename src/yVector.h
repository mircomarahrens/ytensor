namespace yy {
    class yVector {
    public:
        yVector();
        ~yVector();
        void push_back(int value);
        int size();
        int operator[](int index);
    };
}
