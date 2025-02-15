#include <gtest/gtest.h>

#define TEST_FRIENDS                                                                                                   \
    friend class yTensorTest_flatten1_Test;                                                                            \
    friend class yTensorTest_flatten2_Test;                                                                            \
    friend class yTensorTest_flatten3_Test;                                                                            \
    friend class yTensorTest_transpose2_Test;

#include "yTensor.h"

#include <complex>
#include <random>

class yTensorTest : public testing::Test
{
    yTensorTest() = default;

    ~yTensorTest() override = default;
};

TEST(DISABLED_yTensorTest, permutation)
{
    std::vector<int> s = {4, 3, 8};
    for (int i : s)
    {
        for (int j = 0; j < i; j++)
            std::cout << j << std::endl;
    }
}

TEST(yTensorTest, flatten1)
{
    std::vector<std::size_t> s = {2, 3, 2};
    yTensor<double> A({2, 3, 2});

    ASSERT_EQ(A.flatten(0, 0, 0), 0);
    ASSERT_EQ(A.flatten(0, 0, 1), 1);
    ASSERT_EQ(A.flatten(0, 1, 0), 2);
    ASSERT_EQ(A.flatten(0, 1, 1), 3);
    ASSERT_EQ(A.flatten(0, 2, 0), 4);
    ASSERT_EQ(A.flatten(0, 2, 1), 5);
    ASSERT_EQ(A.flatten(1, 0, 0), 6);
    ASSERT_EQ(A.flatten(1, 0, 1), 7);
    ASSERT_EQ(A.flatten(1, 1, 0), 8);
    ASSERT_EQ(A.flatten(1, 1, 1), 9);
    ASSERT_EQ(A.flatten(1, 2, 0), 10);
    ASSERT_EQ(A.flatten(1, 2, 1), 11);
}

TEST(yTensorTest, flatten2)
{
    std::vector<std::size_t> s = {2, 3, 2};
    std::vector<int> d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    yTensor<int> A(d, {2, 3, 2});

    ASSERT_EQ(A.flatten(0, 0, 0), 0);
    ASSERT_EQ(A.flatten(0, 0, 1), 1);
    ASSERT_EQ(A.flatten(0, 1, 0), 2);
    ASSERT_EQ(A.flatten(0, 1, 1), 3);
    ASSERT_EQ(A.flatten(0, 2, 0), 4);
    ASSERT_EQ(A.flatten(0, 2, 1), 5);
    ASSERT_EQ(A.flatten(1, 0, 0), 6);
    ASSERT_EQ(A.flatten(1, 0, 1), 7);
    ASSERT_EQ(A.flatten(1, 1, 0), 8);
    ASSERT_EQ(A.flatten(1, 1, 1), 9);
    ASSERT_EQ(A.flatten(1, 2, 0), 10);
    ASSERT_EQ(A.flatten(1, 2, 1), 11);

    ASSERT_EQ(A(0, 0, 0), d[0]);
    ASSERT_EQ(A(0, 0, 1), d[1]);
    ASSERT_EQ(A(0, 1, 0), d[2]);
    ASSERT_EQ(A(0, 1, 1), d[3]);
    ASSERT_EQ(A(0, 2, 0), d[4]);
    ASSERT_EQ(A(0, 2, 1), d[5]);
    ASSERT_EQ(A(1, 0, 0), d[6]);
    ASSERT_EQ(A(1, 0, 1), d[7]);
    ASSERT_EQ(A(1, 1, 0), d[8]);
    ASSERT_EQ(A(1, 1, 1), d[9]);
    ASSERT_EQ(A(1, 2, 0), d[10]);
    ASSERT_EQ(A(1, 2, 1), d[11]);
}

TEST(yTensorTest, flatten3)
{
    std::vector<std::size_t> s = {16, 4, 4};
    yTensor<double> A(s);

    std::unordered_map<std::size_t, std::vector<std::size_t>> indices = {};

    std::size_t index = 0;
    for (std::size_t i0 = 0; i0 < s[0]; ++i0)
        for (std::size_t i1 = 0; i1 < s[1]; ++i1)
            for (std::size_t i2 = 0; i2 < s[2]; ++i2)
            {
                indices[index] = std::vector<std::size_t>({i0, i1, i2});
                index += 1;
            }

    for (const auto &[key, val] : indices)
    {
        ASSERT_EQ(A.flatten(val), key);
    }
}

TEST(yTensorTest, constructorShape)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<double> A({4, 3, 8});
    ASSERT_EQ(A.shape(), s);
}

TEST(yTensorTest, constructorRandomData)
{
    std::vector<std::size_t> s = {4, 3, 1, 8};

    std::vector<double> data(4 * 3 * 8);

    std::default_random_engine gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(std::begin(data), std::end(data), [&] { return dist(gen); });

    yTensor<double> A(data, {4, 3, 1, 8});
    ASSERT_EQ(A.shape(), s);
    ASSERT_EQ(A.getData(), data);
    ASSERT_EQ(A.getData().size(), 4 * 3 * 8);
}

TEST(yTensorTest, randomize)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    A.randomize();
    ASSERT_EQ(A.shape(), s);
    ASSERT_EQ(A.getData().size(), 4 * 3 * 8);
}

TEST(yTensorTest, dimension)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    ASSERT_EQ(A.dimension(), 3);
}

TEST(yTensorTest, expand_dims)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});

    A.expand_dims(0);
    std::vector<std::size_t> new_shape1 = {1, 4, 3, 8};
    ASSERT_EQ(A.dimension(), 4);
    ASSERT_EQ(A.shape(), new_shape1);

    A.expand_dims(4);
    std::vector<std::size_t> new_shape2 = {1, 4, 3, 8, 1};
    ASSERT_EQ(A.dimension(), 5);
    ASSERT_EQ(A.shape(), new_shape2);

    A.expand_dims(2);
    std::vector<std::size_t> new_shape3 = {1, 4, 1, 3, 8, 1};
    ASSERT_EQ(A.dimension(), 6);
    ASSERT_EQ(A.shape(), new_shape3);
}

TEST(yTensorTest, reshapeMatch)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    A.reshape({3, 4, 2, 2, 2});
}

TEST(yTensorTest, reshapeNoMatch)
{
    std::vector<int> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    EXPECT_THROW(A.reshape({8, 8}), std::logic_error);
}

TEST(yTensorTest, accessOperator)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    A.randomize();
    ASSERT_EQ(A.shape(), s);
    ASSERT_EQ(A.getData().size(), 4 * 3 * 8);

    std::complex<double> val(2.0, 2.0);
    A(1, 2, 4) = val;
    ASSERT_EQ(A(1, 2, 4), val);
}

TEST(yTensorTest, accessOperator_IndexOutOfRange)
{
    std::vector<std::size_t> s = {4, 3, 8};
    yTensor<std::complex<double>> A({4, 3, 8});
    A.randomize();
    ASSERT_EQ(A.shape(), s);
    ASSERT_EQ(A.getData().size(), 4 * 3 * 8);

    EXPECT_THROW(A(1, 4, 4), std::logic_error);
}

TEST(yTensorTest, asterikOperator_int)
{
    std::vector<std::size_t> s = {3, 1, 2};
    std::vector<int> d = {1, 2, 3, 4, 5, 6};
    yTensor<int> A(d, s);

    auto A1 = A * 2;
    auto A2 = 2 * A;
}

TEST(yTensorTest, prod1)
{
    std::vector<std::size_t> s = {3};
    std::vector<int> d = {1, 2, 3};
    yTensor<int> A(d, s);

    auto res = A.prod(0);

    ASSERT_EQ(res, 6);
}

TEST(yTensorTest, prod2)
{
    std::vector<std::size_t> s = {3, 1, 2};
    std::vector<int> d = {1, 2, 3, 4, 5, 6};
    yTensor<int> A(d, s);

    auto res1 = A.prod(1);
    ASSERT_EQ(res1, 6);

    auto res2 = A.prod(2);
    ASSERT_EQ(res2, 5 * 6);
}

TEST(yTensorTest, prod3)
{
    std::vector<std::size_t> s = {3, 2, 1, 3, 4};
    std::vector<int> d(72);
    std::iota(d.begin(), d.end(), 1);
    yTensor<int> A(d, s);

    auto res = A.prod({1, 3});

    ASSERT_EQ(res[0], A(2, 0, 0, 2, 3) * A(2, 1, 0, 2, 3));
    ASSERT_EQ(res[1], A(2, 1, 0, 0, 3) * A(2, 1, 0, 1, 3) * A(2, 1, 0, 2, 3));
}

TEST(yTensorTest, transpose1)
{
    std::vector<std::size_t> s = {8, 7, 6};
    std::vector<int> d(336);
    std::iota(d.begin(), d.end(), 0);
    yTensor<int> A(d, s);
    auto before = A(1, 2, 3);
    A.transpose(std::vector<std::size_t>({2, 0, 1}));
    ASSERT_EQ(before, A(3, 1, 2));
}

TEST(yTensorTest, transpose2)
{
    std::vector<std::size_t> s = {16, 4, 4};
    std::vector<int> d(256);
    std::iota(d.begin(), d.end(), 0);
    yTensor<int> A(d, s);

    std::unordered_map<std::size_t, std::vector<std::size_t>> indices = {};

    std::size_t index = 0;
    for (std::size_t i0 = 0; i0 < s[0]; ++i0)
        for (std::size_t i1 = 0; i1 < s[1]; ++i1)
            for (std::size_t i2 = 0; i2 < s[2]; ++i2)
            {
                indices[index] = std::vector<std::size_t>({i0, i1, i2});
                index += 1;
            }

    for (const auto &[key, val] : indices)
    {
        ASSERT_EQ(key, A.flatten(val));
    }

    A.transpose(std::vector<std::size_t>({2, 0, 1}));

    // TODO transform indices
}

// TEST(DISABLED_TensorTest, dot) {
// //    Tensor<std::complex<double>> A({4, 3, 2, 8});
// //    A.randomize();
// //
// //    Tensor<std::complex<double>> B({8, 4, 2, 1, 3});
// //    B.randomize();
// //
// //    auto C = nlinalg::dot(A, B);
// }

// TEST(DISABLED_TensorTest, tensordot) {
//     // Shape shapeA = {4, 3, 2, 8};
//     // Tensor<std::complex<double>> A(shapeA);

//     // Shape shapeB = {8, 4, 2, 1, 3};
//     // Tensor<std::complex<double>> B(shapeB);

//     // auto C = xt::linalg::tensordot(A, B, {0,2,3}, {1,2,0});
// }
