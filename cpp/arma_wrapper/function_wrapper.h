/*
 * This header file contains code that allows you to call a function on each of a function's arguments and the 
 * values that it returns. 
 */ 

#ifndef FUNCTION_WRAPPER_H
#define FUNCTION_WRAPPER_H

#include <type_traits>
#include <tuple>
#include <utility>
#include <functional>
#include "arma_wrapper.h"


namespace aw {


/************************************ type_traits style classes *************************************/ 

/* Tests if U is a specialization of T */
template<template<typename...> typename T, typename U> 
struct is_specialization_of : std::false_type {};


template<template <typename ...> typename T, typename... Args>
struct is_specialization_of<T, T<Args...>> : std::true_type {};   


/* For generic types use the type signature of their operator() */
template <typename T>
struct closure_traits : public closure_traits<decltype(&T::operator())> {}; 


/* 
 * This is adapted from the stack overflow question 
 * Is it possible to figure out the parameter type and return type of a lambda. 
 */
template <typename ClassType, typename ReturnType, typename... ArgTypes>                        
struct closure_traits<ReturnType (ClassType::*) (ArgTypes... args) const>                  
{                                                                          
    using arity = std::integral_constant<std::size_t, sizeof...(ArgTypes)>;   
    using Ret = ReturnType;                                                 
    
    template <std::size_t I>
    struct Args {
        using type = typename std::tuple_element<I, std::tuple<ArgTypes...>>::type; 

    };
    
};


/* I  am not sure how to get the top part to accept functions were this is not const. */
template <typename ClassType, typename ReturnType, typename... ArgTypes>                        
struct closure_traits<ReturnType (ClassType::*) (ArgTypes... args)>                  
{                                                                          
    using arity = std::integral_constant<std::size_t, sizeof...(ArgTypes)>;   
    using Ret = ReturnType;                                                 
    
    template <std::size_t I>
    struct Args {
        using type = typename std::tuple_element<I, std::tuple<ArgTypes...>>::type; 

    };
    
};

/************************** Implementation Details ********************************/

namespace detail {

    
    /* 
     * This function defines a generic lambda that takes can be applied to every one of the arguments in the 
     * tuple. It requires that Func has a overload for operator() that can be applied to each of the parameters. 
     * Then it applies this lambda to each of the parameters in the tuple. 
     */ 
    template<typename Func, typename TupleType, std::size_t... I> 
    decltype(auto) for_each_impl(TupleType&& tup, std::index_sequence<I...>) {
    
        auto func_impl = [] (auto&& x) { 
            Func func;
            return func(std::forward<decltype(x)>(x));
        };
        
        return std::make_tuple(func_impl(std::get<I>(std::forward<TupleType>(tup)))...); 
    
    }
    
    
    /* My version of c++17 apply_impl method. */ 
    template<typename FuncType, typename TupleType, std::size_t... I>
    decltype(auto) apply_impl(FuncType&& func, TupleType&& tup, std::index_sequence<I...>) {
    
        return func(std::get<I>(std::forward<TupleType>(tup))...); 
    
    }

}



/***************************** Various Helper Functions *********************************/

/* 
 * These two functions convert the argument to a tuple if it is not one already so that the caller can assume
 * he is dealing with a tuple. 
 */
template<typename T> 
auto idempotent_make_tuple(T&& arg)  -> std::enable_if_t<is_specialization_of<std::tuple,T>::value, T&&> { 
   
    return arg;  

}


template<typename T>
auto idempotent_make_tuple(T&& arg)  -> std::enable_if_t<! is_specialization_of<std::tuple, T>::value, 
                                            decltype(std::make_tuple(std::forward<T>(arg)))>  { 

    return std::make_tuple(std::forward<T>(arg));  

}



/* This is just a slightly less generic version of c++17 std::apply function. */
template<typename FuncType, typename TupleType> 
decltype(auto) apply(FuncType&& func, TupleType&& tup) {

    return detail::apply_impl(std::forward<FuncType>(func), std::forward<TupleType>(tup),
                      std::make_index_sequence<std::tuple_size<std::decay_t<TupleType>>::value>{}); 

}


/* Applies a Functor Func to each element of the tuple. As you might expect, the Functor needs overloads
 * for all of the types that in the tuple or the code will not compile. 
 */
template<typename Func, typename TupleType> 
decltype(auto) for_each_in_tuple(TupleType&& tup) { 

    return  detail::for_each_impl<Func>(std::forward<TupleType>(tup), 
                             std::make_index_sequence<std::tuple_size<std::decay_t<TupleType>>::value>{}); 

}




/******************* More implementation Details **************************/

namespace detail {

    /* 
     * This function takes a function and an index sequence with its number of arguments. It then figures out 
     * the types of its arguments, and creates a new function with each of the arguments and each of the returned
     * values converted to the new types.   
     */ 
    template<typename ArgWrapper, typename ReturnWrapper, typename FuncType, size_t...I>
    auto wrap_impl(FuncType&& func, std::index_sequence<I...>) {
    
        /* This is used to figure out what the argument types of func are */
        using traits = closure_traits<typename std::decay_t<FuncType>>; 
    
        auto wrapped_func = [=] (std::result_of_t<ReturnWrapper(
            typename traits:: template Args<I>::type)>... args) { 

            /* Apply the argument wrapper function to each of the arguments of the new function. */ 
            decltype(auto) tup1 = for_each_in_tuple<ArgWrapper>(std::forward_as_tuple(args...));  
            /* Apply the old function to the wrapped arguments. */
            decltype(auto) tup2 = idempotent_make_tuple(apply(func, 
                std::forward<std::decay_t<decltype(tup1)>>(tup1))); 
            /* Apply the Return wrapper to the return value of the old function */
            decltype(auto) tup3 = for_each_in_tuple<ReturnWrapper>(
                std::forward<std::decay_t<decltype(tup2)>>(tup2)); 
    
            return tup3; 
        };
    
        return wrapped_func; 
    
    }


}


/****************************** The main method *********************************/

/* 
 * This function takes a function as an argument and applies a converter function to each of its arguments 
 * as well as of the values that it returns. 
 */ 
template<typename ArgWrapper = make_cpp, typename ReturnWrapper = make_py, typename FuncType>
auto wrap(FuncType&& func) {

    return detail::wrap_impl<ArgWrapper, ReturnWrapper>(
               std::forward<FuncType>(func), std::make_index_sequence<closure_traits<FuncType>::arity::value> {});  



}


}


#endif /* FUNCTION_WRAPPER_H */
