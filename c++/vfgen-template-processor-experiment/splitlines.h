#pragma once

#include <string>
#include <vector>

inline std::vector<std::string>
splitlines(const std::string text, const std::string linesep)
{
    std::vector<std::string> lines;
    size_t current = 0;
    do {
        size_t loc = text.find(linesep, current);
        if (loc == text.npos) {
            loc = text.size();
        }
        lines.push_back(text.substr(current, loc - current));
        current = loc + linesep.size();
    } while (current < text.size());
    return lines;
}
