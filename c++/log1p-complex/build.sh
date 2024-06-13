unamestr=$(uname)
if [[ "$unamestr" == 'Darwin' ]]; then
   EXTRA_CXX_FLAGS=-mmacosx-version-min=13.3 make
else
   make
fi
