cmake_minimum_required(VERSION 3.10)
project(SDOF)

set(CMAKE_CXX_STANDARD 17)

find_package(Allegro5 REQUIRED COMPONENTS Main Primitives Font TTF)

include_directories(include)

file(GLOB SOURCES 
    "src/*.cpp"
)

add_executable(SDOF ${SOURCES})

target_link_libraries(SDOF 
    PRIVATE 
    Allegro::Allegro 
    Allegro::ACodec 
    Allegro::APrimitives 
    Allegro::AFont 
    Allegro::ATTF
)

install(FILES resources/VeraBd.ttf DESTINATION ${CMAKE_BINARY_DIR})