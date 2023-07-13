target_sources(TestFlextopFD
    PRIVATE
        test_flextop.cpp
)

target_sources(TimingTest
    PRIVATE
        test_time.cpp
)  

INSTALL(TARGETS TestFlextopFD DESTINATION "${CMAKE_INSTALL_PREFIX}/tests")
INSTALL(TARGETS TimingTest DESTINATION "${CMAKE_INSTALL_PREFIX}/tests")