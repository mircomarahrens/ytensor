#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    // Initialize Google's testing library.
    ::testing::InitGoogleTest(&argc, argv);

    const int result = RUN_ALL_TESTS();

    return result;
}
