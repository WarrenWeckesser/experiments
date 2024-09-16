#include <cstdio>
#include <set>
#include <vector>
#include "simple_array.h"

using namespace std;

int main()
{
    shape_t shape{3, 3, 2};
    vector<int> data{1, 10, -3, 2, 5, 5,
                     0, 0, -4, 10, 12, 99,
                     2, 2, 3, 3, 4, 4};

    // a has shape (3, 3, 2)
    SimpleArray a{shape, data};

    printf("a.flat_size() = %zu\n", a.flat_size());

    for (size_t i = 0; i < a.flat_size(); ++i) {
        printf("%3zu  %d\n", i, a.flat(i));
    }

    printf("\n");
    MultiIndexIterator ita = a.index_iterator();
    printf("got ita\n");
    for (auto &index: ita) {
        for (auto &i: index) {
            printf(" %3zu", i);
        }
        printf("  %d\n", a[index]);
    }

    // SimpleArray b{shape_t{}, vector<int>{10}};
    SimpleArray b = make_scalar_array(11);

    printf("b.flat_size() = %zu\n", b.flat_size());
    printf("b[{}] = %d\n", b[shape_t{}]);

    SimpleArray c = make_1d_array(vector<string>{"ABC", "DEF", "GHI"});

    printf("\n");
    printf("c.flat_size() = %zu\n", c.flat_size());
    printf("c[{1}] = %s\n", c[multiindex_t{1}].c_str());

    printf("\n");
    MultiIndexIterator itc = c.index_iterator();
    for (auto &index: itc) {
        for (auto &i: index) {
            printf(" %3zu", i);
        }
        printf("  %s\n", c[index].c_str());
    }

    SimpleArray z = make_scalar_array(125);

    printf("\n");
    printf("z.flat_size() = %zu\n", z.flat_size());
    printf("z[{}] = %d\n", z[multiindex_t{}]);

    printf("\n");
    MultiIndexIterator itz = z.index_iterator();
    for (auto &index: itz) {
        for (auto &i: index) {
            printf(" %3zu", i);
        }
        printf("  %d\n", z[index]);
    }

    printf("\n>\n");
    MultiIndexIterator it{shape_t{2, 3, 2}};
    for (auto &index: it) {
        printf("(");
        for (auto &i: index) {
            printf(" %3zu", i);
        }
        printf(")\n");
    }

    printf("testing broadcast_shape()\n");
    SimpleArray q0 = make_scalar_array(99);
    SimpleArray q1 = make_1d_array(vector<int>{10, 11});
    SimpleArray q2 = SimpleArray{shape_t{3, 1}, vector<int>{-1, -2, -3}};
    vector<SimpleArray<int>> arrays{q0, q1, q2};
    shape_t bc_shape = broadcast_shape(arrays);
    printf("bc_shape:");
    for (auto &len: bc_shape) {
        printf("  %zu", len);
    }
    printf("\n");

    printf("q2._shape:         ");
    for (auto &s: q2._shape) {
        printf(" %3zu", s);
    }
    printf("\n");
    printf("q2._index_strides: ");
    for (auto &s: q2._index_strides) {
        printf(" %3zu", s);
    }
    printf("\n");
    q2.dump("q2");
    SimpleArray q2bc = broadcast_to(q2, shape_t{3, 2});
    q2bc.dump("q2bc");
    MultiIndexIterator itq2bc = q2bc.index_iterator();
    for (auto &index: itq2bc) {
        for (auto &i: index) {
            printf(" %3zu", i);
        }
        printf("  %d\n", q2bc[index]);
    }

    b.dump("b");
    SimpleArray bc = broadcast_to(b, shape_t{3});
    bc.dump("bc");

    // MultiIndexIterator it = MultiIndexIterator(shape);
    // for (auto &index: it) {
    //     for (auto &i: index) {
    //         printf(" %3zu", i);
    //     }
    //     printf("  %d\n", a[index]);
    // }

    // printf("\n");
    // MultiIndexIterator it2 = a.index_iterator();
    // for (auto index = it2.begin(); index != it2.end(); ++index) {
    //     for (auto &i: *index) {
    //         printf(" %3zu", i);
    //     }
    //     printf("  %d\n", a[*index]);
    // }
}
