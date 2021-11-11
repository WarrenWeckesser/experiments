//
// Use libcurl to read the contents of a URL into a C++ string.
//
// This file started out as the example "getinmemory.c" from the libcurl
// web site.
//

#include <iostream>
#include <string>
#include <curl/curl.h>


static size_t
build_string_callback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    std::string *str = (std::string *) userp;

    (*str).append((char *) contents, realsize);

    return realsize;
}


int main ()
{
    std::string str;

    CURL *curl_handle;
    CURLcode res;

    // Boilerplate libcurl initialization.
    curl_global_init(CURL_GLOBAL_ALL);

    // Initialize the curl session.
    curl_handle = curl_easy_init();

    // Specify URL to retrieve.
    curl_easy_setopt(curl_handle, CURLOPT_URL, "http://www.example.com/");

    // Send all data to the function build_string_callack().
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, build_string_callback);

    // We pass our string to the callback function.
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *) &str);

    // Some servers don't like requests that are made without a user-agent
    // field, so we provide one.
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");

    // Get it the data.
    res = curl_easy_perform(curl_handle);

    // Check for errors.
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: %s\n" << curl_easy_strerror(res);
    }
    else {
        // The URL contents are now in str.
        std::cout << "Here it is.........\n";
        std::cout << str << std::endl;
    }

    // Cleanup curl stuff.
    curl_easy_cleanup(curl_handle);

    // Done with libcurl.
    curl_global_cleanup();

    return 0;
}
