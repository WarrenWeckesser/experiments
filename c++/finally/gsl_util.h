#ifndef GSL_UTIL_H_
#define GSL_UTIL_H_

// From https://github.com/microsoft/GSL/blob/543d0dd3fe966ddf20e884b44e5fdbf12cb43784/include/gsl/util
// That code is released under the MIT license:
//
// Copyright (c) 2015 Microsoft Corporation. All rights reserved. 
//
// This code is licensed under the MIT License (MIT). 
//
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
// of the Software, and to permit persons to whom the Software is furnished to do 
// so, subject to the following conditions: 
//
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software. 
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE. 


#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define GSL_NODISCARD [[nodiscard]]
#else
#define GSL_NODISCARD
#endif // defined(__cplusplus) && (__cplusplus >= 201703L)


template <class F>
class final_action
{
public:
    explicit final_action(const F& ff) noexcept : f{ff} { }
    explicit final_action(F&& ff) noexcept : f{std::move(ff)} { }

    ~final_action() noexcept { if (invoke) f(); }

    final_action(final_action&& other) noexcept
        : f(std::move(other.f)), invoke(std::exchange(other.invoke, false))
    { }

    final_action(const final_action&)   = delete;
    void operator=(const final_action&) = delete;
    void operator=(final_action&&)      = delete;

private:
    F f;
    bool invoke = true;
};

// finally() - convenience function to generate a final_action
template <class F>
GSL_NODISCARD auto finally(F&& f) noexcept
{
    return final_action<std::decay_t<F>>{std::forward<F>(f)};
}

#endif   // GSL_UTIL_H_
