import { Statistics, test, summary } from "./test.js";
import { Shared, Vector, Matrix, Cost, Activations, NeuralNet } from "./index.js";

const stats = new Statistics();

console.log(`Suite 0: Validation Functions`);

test("throws error when one vector is not a Float64Array", stats, () => {
    try {
        Shared.validateStructure(undefined, new Float64Array(5));

        return false;
    } catch(e) {
        return e.message === "Both vectors must be a Float64Array";
    }
});
test("throws error when one vector has zero length", stats, () => {
    try {
        Shared.validateStructure(new Float64Array(12), new Float64Array(0));

        return false;
    } catch(e) {
        return e.message === "Both vectors must have positive length";
    }
});
test("throws error when both vector lengths don't match", stats, () => {
    try {
        Shared.validateSameLength(new Float64Array(3), new Float64Array(5));

        return false;
    } catch(e) {
        return e.message === "Both vectors must have the same length for this operation";
    }
});
test("throws error when vector is not a Float64Array", stats, () => {
    try {
        Shared.validateStructureSingle(undefined);

        return false;
    } catch(e) {
        return e.message === "Vector must be a Float64Array";
    }
});
test("throws error when vector has zero length", stats, () => {
    try {
        Shared.validateStructureSingle(new Float64Array(0));

        return false;
    } catch(e) {
        return e.message === "Vector must have positive length";
    }
});
test("throws error when a matrix is not a 2D Float64Array", stats, () => {
    try {
        Shared.validateStructureMat([
            new Float64Array(2),
            new Float64Array(2),
        ], undefined);

        return false;
    } catch(e) {
        return e.message === "Both matrices must be a 2D Float64Array";
    }
});
test("throws error when a matrix has no rows", stats, () => {
    try {
        Shared.validateStructureMat(new Array(), [
            new Float64Array(2),
            new Float64Array(2),
        ]);

        return false;
    } catch(e) {
        return e.message === "Both matrices must be a 2D Float64Array";
        // in row major, zero rows mean no nested arrays
        // no nested arrays -> 1D array
        // thus, the thrown error will always be the one above
    }
});
test("throws error when a matrix has zero-length cols", stats, () => {
    try {
        Shared.validateStructureMat([
            new Float64Array(0),
            new Float64Array(0),
        ], [
            new Float64Array(3),
            new Float64Array(3),
        ]);
        
        return false;
    } catch(e) {
        return e.message === "Both matrices must have positive row and column length";
    }
});
test("throws error when both matrices' dimensions don't match", stats, () => {
    try {
        Shared.validateSameShape([
            new Float64Array(3),
            new Float64Array(3),
        ], [
            new Float64Array(2),
            new Float64Array(2),
        ]);

        return false;
    } catch(e) {
        return e.message === "Both matrices must have the same dimensions for this operation";
    }
});

console.log("");
console.log("Suite 1: Vector Class");

console.log(`  Group 1: vector creation`);
test("creates a vector of length n", stats, () => {
    return Vector.create(10, -0.5, 0.5).length === 10;
});
test("throws error when parameter is 0", stats, () => {
    try {
        Vector.create(0, -0.5, 0.5);

        return false;
    } catch(e) {
        return e.message === "Length parameter must be a positive number";
    }
});
test("throws error when parameter is negative", stats, () => {
    try {
        Vector.create(-5, -0.5, 0.5);

        return false;
    } catch(e) {
        return e.message === "Length parameter must be a positive number";
    }
});
test("throws error when parameter is not a number", stats, () => {
    try {
        Vector.create(undefined, -0.5, 0.5);

        return false;
    } catch(e) {
        return e.message === "Length parameter must be a positive number";
    }
});
test("throws error when a range limit is not a number", stats, () => {
    try {
        Vector.create(10, -0.5, undefined);

        return false;
    } catch(e) {
        return e.message === "Both range limits must be a number";
    }
})

console.log(`  Group 2: vector operations`);
test("adds two vectors together", stats, () => {
    const a = new Float64Array([2, 1]);
    const b = new Float64Array([5, 4]);
    const expected = [
        a[0] + b[0],
        a[1] + b[1],
    ]
    const result = Vector.add(a, b);
    return (
        expected[0] === result[0] &&
        expected[1] === result[1]
    );
});
test("subtracts a vector from another", stats, () => {
    const a = new Float64Array([9, 2, 16]);
    const b = new Float64Array([6, 1, 8]);
    const expected = [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    ]
    const result = Vector.subtract(a, b);
    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the dot product", stats, () => {
    const a = new Float64Array([2, 7, 6, 0, 1]);
    const b = new Float64Array([4, 1, 2, 5, 6]);
    const expected = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4];
    const result = Vector.dot(a, b);
    return expected === result;
});
test("computes the elementwise product", stats, () => {
    const a = new Float64Array([9, 26, 22]);
    const b = new Float64Array([6, 1, 32]);
    const expected = [
        a[0] * b[0],
        a[1] * b[1],
        a[2] * b[2],
    ]
    const result = Vector.elementwise(a, b);
    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the outer product", stats, () => {
    const a = new Float64Array([4, 1]);
    const b = new Float64Array([1, 6, 99, 2]);
    const expected = [
        [a[0] * b[0], a[0] * b[1], a[0] * b[2], a[0] * b[3]],
        [a[1] * b[0], a[1] * b[1], a[1] * b[2], a[1] * b[3]],
    ];
    const result = Vector.outer(a, b);
    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[0][3] === result[0][3] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2] &&
        expected[1][3] === result[1][3]
    );
});

console.log("");
console.log("Suite 2: Matrix Class");

console.log(`  Group 1: matrix creation`);
test("creates a row-major matrix of size rows * cols", stats, () => {
    const mat = Matrix.create(2, 4, -0.5, 0.5);
    return (
        mat.length === 2 &&
        mat[0].length === 4
    );
});
test("throws error when a row/col parameter is 0", stats, () => {
    try {
        Matrix.create(2, 0, -0.5, 0.5);
 
        return false;
    } catch(e) {
        return e.message === "Both parameters must be a positive number";
    }
});
test("throws error when a row/col parameter is negative", stats, () => {
    try {
        Matrix.create(-20, 1, -0.5, 0.5);

        return false;
    } catch(e) {
        return e.message === "Both parameters must be a positive number";
    }
});
test("throws error when a row/col parameter is not a number", stats, () => {
    try {
        Matrix.create(undefined, 6, -0.5, 0.5);

        return false;
    } catch(e) {
        return e.message === "Both parameters must be a positive number";
    }
});
test("throws error when a range limit is not a number", stats, () => {
    try {
        Matrix.create(4, 6, undefined, 0.5);

        return false;
    } catch(e) {
        return e.message === "Both range limits must be a number";
    }
});

console.log(`  Group 2: matrix operations`);
test("adds two matrices together", stats, () => {
    const a = [
        new Float64Array([56, 2]),
        new Float64Array([21, 12]),
    ];
    const b = [
        new Float64Array([74, 4]),
        new Float64Array([3, 9]),
    ];
    const expected = [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ];
    const result = Matrix.add(a, b);
    
    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1]
    );
});
test("subtracts a matrix from another", stats, () => {
    const a = [
        new Float64Array([22, 0]),
        new Float64Array([7, 9]),
    ];
    const b = [
        new Float64Array([72, 1]),
        new Float64Array([4, 6]),
    ];
    const expected = [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ];
    const result = Matrix.subtract(a, b);
    
    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1]
    );
});
test("computes the elementwise product", stats, () => {
    const a = [
        new Float64Array([56, 2]),
        new Float64Array([21, 12]),
    ];
    const b = [
        new Float64Array([74, 4]),
        new Float64Array([3, 9]),
    ];
    const expected = [
        [a[0][0] * b[0][0], a[0][1] * b[0][1]],
        [a[1][0] * b[1][0], a[1][1] * b[1][1]],
    ];
    const result = Matrix.elementwise(a, b);
    
    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1]
    );
});
test("transposes a matrix", stats, () => {
    const a = [
        new Float64Array([56, 2, 4]),
        new Float64Array([21, 12, 68]),
    ];
    const expected = [
        [a[0][0], a[1][0]],
        [a[0][1], a[1][1]],
        [a[0][2], a[1][2]],
    ]
    const result = Matrix.transpose(a);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[2][0] === result[2][0] &&
        expected[2][1] === result[2][1]
    );
});
test("adds a scalar to a matrix", stats, () => {
    const m = [
        new Float64Array([2, 3]),
        new Float64Array([1, 6]),
    ];
    const s = 6;
    const expected = [
        [m[0][0] + s, m[0][1] + s],
        [m[1][0] + s, m[1][1] + s],
    ];
    const result = Matrix.addScalar(m, s);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1]
    );
});
test("flips a matrix across both axes", stats, () => {
    const mat = [
        new Float64Array([2, 3, 2]),
        new Float64Array([0, 1, 4]),
        new Float64Array([9, 8, 6]),
    ];
    const expected = [
        [mat[2][2], mat[2][1], mat[2][0]],
        [mat[1][2], mat[1][1], mat[1][0]],
        [mat[0][2], mat[0][1], mat[0][0]],
    ];
    const result = Matrix.flip(mat);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2] &&
        expected[2][0] === result[2][0] &&
        expected[2][1] === result[2][1] &&
        expected[2][2] === result[2][2]
    );
});
test("flattens a matrix into a vector", stats, () => {
    const mat = [
        new Float64Array([5, 3]),
        new Float64Array([2, 4]),
    ];
    const expected = [
        mat[0][0],
        mat[0][1],
        mat[1][0],
        mat[1][1],
    ];
    const result = Matrix.flatten(mat);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2] &&
        expected[3] === result[3]
    )
});
test("reshapes a vector into a matrix", stats, () => {
    const vec = new Float64Array([2, 3, 1, 9, 8, 4]);
    const expected = [
        [vec[0], vec[1], vec[2]],
        [vec[3], vec[4], vec[5]],
    ];
    const result = Matrix.reshape(vec, 2, 3);
    
    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2]
    );
});

console.log(`  Group 3: matrix multiplication`);
test("multiplies two matrices together", stats, () => {
    const a = [
        new Float64Array([5, 6]),
        new Float64Array([12, 0]),
        new Float64Array([5, 2]),
    ];
    const b = [
        new Float64Array([12, 54, 6]),
        new Float64Array([83, 22, 31]),
    ];
    const expected = [
        [
            a[0][0]*b[0][0] + a[0][1]*b[1][0], 
            a[0][0]*b[0][1] + a[0][1]*b[1][1], 
            a[0][0]*b[0][2] + a[0][1]*b[1][2],
        ],
        [
            a[1][0]*b[0][0] + a[1][1]*b[1][0], 
            a[1][0]*b[0][1] + a[1][1]*b[1][1], 
            a[1][0]*b[0][2] + a[1][1]*b[1][2],
        ],
        [
            a[2][0]*b[0][0] + a[2][1]*b[1][0], 
            a[2][0]*b[0][1] + a[2][1]*b[1][1], 
            a[2][0]*b[0][2] + a[2][1]*b[1][2],
        ],
    ]
    const result = Matrix.multiply(a, b);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2] &&
        expected[2][0] === result[2][0] &&
        expected[2][1] === result[2][1] &&
        expected[2][2] === result[2][2]
    );
});
test("throws error when columns of A is not equal to rows of B", stats, () => {
    try {
        const a = [
            new Float64Array([52, 1]),
            new Float64Array([1, 0]),
            new Float64Array([35, 22]),
        ];
        const b = [
            new Float64Array([0, 4]),
            new Float64Array([3, 62]),
            new Float64Array([5, 1])
        ];

        Matrix.multiply(a, b);

        return false;
    } catch(e) {
        return e.message === "The columns of A must be equal to the rows of B\nfor matrix multiplication.\nTry transposing a matrix or flipping the order of arguments";
    }
});

console.log(`  Group 4: matrix vector multiplication`);
test("multiplies a matrix and vector", stats, () => {
    const m = [
        new Float64Array([2, 4, 5]),
        new Float64Array([1, 5, 2]),
    ];
    const v = new Float64Array([4, 2, 3]);
    const expected = [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
    ];
    const result = Matrix.matrixVector(m, v);
    
    return (
        expected[0] === result[0] &&
        expected[1] === result[1]
    );
});
test("throws error when vector is not a Float64Array", stats, () => {
    try {
        const m = [
            new Float64Array([2, 2]), 
            new Float64Array([2, 2])
        ];
        Matrix.matrixVector(m, undefined);

        return false;
    } catch(e) {
        return e.message === "Vector must be a Float64Array";
    }
});
test("throws error when vector has zero length", stats, () => {
    try {
        const m = [
            new Float64Array([2, 2]), 
            new Float64Array([2, 2])
        ];
        Matrix.matrixVector(m, new Float64Array(0));

        return false;
    } catch(e) {
        return e.message === "Vector must have positive length";
    }
});
test("throws error when matrix is not a 2D Float64Array", stats, () => {
    try {
        const v = new Float64Array([2, 2]);
        Matrix.matrixVector(undefined, v);

        return false;
    } catch(e) {
        return e.message === "Matrix must be a 2D Float64Array";
    }
});
test("throws error when matrix has no rows", stats, () => {
    try {
        const v = new Float64Array([2, 2]);
        Matrix.matrixVector([], v);

        return false;
    } catch(e) {
        return e.message === "Matrix must be a 2D Float64Array";
        // for elaboration on error message, check ln 75
    }
});
test("throws error when matrix has zero-length cols", stats, () => {
    try {
        const v = new Float64Array([2, 2]);
        Matrix.matrixVector([
            new Float64Array(0),
            new Float64Array(0),
        ], v);

        return false;
    } catch(e) {
        return e.message === "Matrix must have positive row and column length";
    }
});
test("throws error when vector length is not equal to matrix column length", stats, () => {
    try {
        const m = [
            new Float64Array([2, 5, 7]),
            new Float64Array([0, 9, 1]),
        ];
        const v = new Float64Array([2, 2]);
        Matrix.matrixVector(m, v);

        return false;
    } catch(e) {
        return e.message === "Vector length must be the same as matrix column length\nfor matrix vector multiplication";
    }
});

console.log(`  Group 5: convolution`);
test("throws error if kernel is larger than src in cross-correlation", stats, () => {
    const k = [
        new Float64Array([1, 0.2, 0.2]),
        new Float64Array([0.8, 0.2, 0.1]),
        new Float64Array([0.4, 0.4, 0.8]),
    ];
    const src = [
        new Float64Array([0, 1]),
        new Float64Array([0, 5]),
    ];
    try {
        Matrix.convolve(src, k);

        return false;
    } catch(e) {
        return e.message === "Kernel cannot have larger size than source matrix";
    }
});
test("cross-correlates a source matrix with a kernel", stats, () => {
    const k = [
        new Float64Array([1, 0.2, 0.2]),
        new Float64Array([0.8, 0.2, 0.1]),
        new Float64Array([0.4, 0.4, 0.8]),
    ];
    const src = [
        new Float64Array([0, 1, 0, 3, 2]),
        new Float64Array([0, 5, 6, 2, 6]),
        new Float64Array([1, 1, 0, 3, 2]),
        new Float64Array([9, 1, 0, 8, 1]),
        new Float64Array([2, 4, 8, 3, 3]),
    ];
    const expected = [
        [
            src[0][0]*k[0][0] + src[0][1]*k[0][1] + src[0][2]*k[0][2]
            + src[1][0]*k[1][0] + src[1][1]*k[1][1] + src[1][2]*k[1][2]
            + src[2][0]*k[2][0] + src[2][1]*k[2][1] + src[2][2]*k[2][2],
            src[0][1]*k[0][0] + src[0][2]*k[0][1] + src[0][3]*k[0][2]
            + src[1][1]*k[1][0] + src[1][2]*k[1][1] + src[1][3]*k[1][2]
            + src[2][1]*k[2][0] + src[2][2]*k[2][1] + src[2][3]*k[2][2],
            src[0][2]*k[0][0] + src[0][3]*k[0][1] + src[0][4]*k[0][2]
            + src[1][2]*k[1][0] + src[1][3]*k[1][1] + src[1][4]*k[1][2]
            + src[2][2]*k[2][0] + src[2][3]*k[2][1] + src[2][4]*k[2][2],
        ],
        [
            src[1][0]*k[0][0] + src[1][1]*k[0][1] + src[1][2]*k[0][2]
            + src[2][0]*k[1][0] + src[2][1]*k[1][1] + src[2][2]*k[1][2]
            + src[3][0]*k[2][0] + src[3][1]*k[2][1] + src[3][2]*k[2][2],
            src[1][1]*k[0][0] + src[1][2]*k[0][1] + src[1][3]*k[0][2]
            + src[2][1]*k[1][0] + src[2][2]*k[1][1] + src[2][3]*k[1][2]
            + src[3][1]*k[2][0] + src[3][2]*k[2][1] + src[3][3]*k[2][2],
            src[1][2]*k[0][0] + src[1][3]*k[0][1] + src[1][4]*k[0][2]
            + src[2][2]*k[1][0] + src[2][3]*k[1][1] + src[2][4]*k[1][2]
            + src[3][2]*k[2][0] + src[3][3]*k[2][1] + src[3][4]*k[2][2],
        ],
        [
            src[2][0]*k[0][0] + src[2][1]*k[0][1] + src[2][2]*k[0][2]
            + src[3][0]*k[1][0] + src[3][1]*k[1][1] + src[3][2]*k[1][2]
            + src[4][0]*k[2][0] + src[4][1]*k[2][1] + src[4][2]*k[2][2],
            src[2][1]*k[0][0] + src[2][2]*k[0][1] + src[2][3]*k[0][2]
            + src[3][1]*k[1][0] + src[3][2]*k[1][1] + src[3][3]*k[1][2]
            + src[4][1]*k[2][0] + src[4][2]*k[2][1] + src[4][3]*k[2][2],
            src[2][2]*k[0][0] + src[2][3]*k[0][1] + src[2][4]*k[0][2]
            + src[3][2]*k[1][0] + src[3][3]*k[1][1] + src[3][4]*k[1][2]
            + src[4][2]*k[2][0] + src[4][3]*k[2][1] + src[4][4]*k[2][2],
        ],
    ];
    const result = Matrix.convolve(src, k);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2] &&
        expected[2][0] === result[2][0] &&
        expected[2][1] === result[2][1] &&
        expected[2][2] === result[2][2]
    );
});
test("performs full mode cross-correlation", stats, () => {
    const k = [
        new Float64Array([1, 0.2, 0.2]),
        new Float64Array([0.8, 0.2, 0.1]),
        new Float64Array([0.4, 0.4, 0.8]),
    ];
    const src = [
        new Float64Array([0, 1, 0, 3, 2]),
        new Float64Array([0, 5, 6, 2, 6]),
        new Float64Array([1, 1, 0, 3, 2]),
        new Float64Array([9, 1, 0, 8, 1]),
        new Float64Array([2, 4, 8, 3, 3]),
    ];
    const padded = [
        [0, 0, 0,         0,         0,         0,         0,         0, 0],
        [0, 0, 0,         0,         0,         0,         0,         0, 0],
        [0, 0, src[0][0], src[0][1], src[0][2], src[0][3], src[0][4], 0, 0],
        [0, 0, src[1][0], src[1][1], src[1][2], src[1][3], src[1][4], 0, 0],
        [0, 0, src[2][0], src[2][1], src[2][2], src[2][3], src[2][4], 0, 0],
        [0, 0, src[3][0], src[3][1], src[3][2], src[3][3], src[3][4], 0, 0],
        [0, 0, src[4][0], src[4][1], src[4][2], src[4][3], src[4][4], 0, 0],
        [0, 0, 0,         0,         0,         0,         0,         0, 0],
        [0, 0, 0,         0,         0,         0,         0,         0, 0],
    ];
    const expected = [
        [
            padded[0][0]*k[0][0] + padded[0][1]*k[0][1] + padded[0][2]*k[0][2]
            + padded[1][0]*k[1][0] + padded[1][1]*k[1][1] + padded[1][2]*k[1][2]
            + padded[2][0]*k[2][0] + padded[2][1]*k[2][1] + padded[2][2]*k[2][2],
            padded[0][1]*k[0][0] + padded[0][2]*k[0][1] + padded[0][3]*k[0][2]
            + padded[1][1]*k[1][0] + padded[1][2]*k[1][1] + padded[1][3]*k[1][2]
            + padded[2][1]*k[2][0] + padded[2][2]*k[2][1] + padded[2][3]*k[2][2],
            padded[0][2]*k[0][0] + padded[0][3]*k[0][1] + padded[0][4]*k[0][2]
            + padded[1][2]*k[1][0] + padded[1][3]*k[1][1] + padded[1][4]*k[1][2]
            + padded[2][2]*k[2][0] + padded[2][3]*k[2][1] + padded[2][4]*k[2][2],
            padded[0][3]*k[0][0] + padded[0][4]*k[0][1] + padded[0][5]*k[0][2]
            + padded[1][3]*k[1][0] + padded[1][4]*k[1][1] + padded[1][5]*k[1][2]
            + padded[2][3]*k[2][0] + padded[2][4]*k[2][1] + padded[2][5]*k[2][2],
            padded[0][4]*k[0][0] + padded[0][5]*k[0][1] + padded[0][6]*k[0][2]
            + padded[1][4]*k[1][0] + padded[1][5]*k[1][1] + padded[1][6]*k[1][2]
            + padded[2][4]*k[2][0] + padded[2][5]*k[2][1] + padded[2][6]*k[2][2],
            padded[0][5]*k[0][0] + padded[0][6]*k[0][1] + padded[0][7]*k[0][2]
            + padded[1][5]*k[1][0] + padded[1][6]*k[1][1] + padded[1][7]*k[1][2]
            + padded[2][5]*k[2][0] + padded[2][6]*k[2][1] + padded[2][7]*k[2][2],
            padded[0][6]*k[0][0] + padded[0][7]*k[0][1] + padded[0][8]*k[0][2]
            + padded[1][6]*k[1][0] + padded[1][7]*k[1][1] + padded[1][8]*k[1][2]
            + padded[2][6]*k[2][0] + padded[2][7]*k[2][1] + padded[2][8]*k[2][2],
        ],
        [
            padded[1][0]*k[0][0] + padded[1][1]*k[0][1] + padded[1][2]*k[0][2]
            + padded[2][0]*k[1][0] + padded[2][1]*k[1][1] + padded[2][2]*k[1][2]
            + padded[3][0]*k[2][0] + padded[3][1]*k[2][1] + padded[3][2]*k[2][2],
            padded[1][1]*k[0][0] + padded[1][2]*k[0][1] + padded[1][3]*k[0][2]
            + padded[2][1]*k[1][0] + padded[2][2]*k[1][1] + padded[2][3]*k[1][2]
            + padded[3][1]*k[2][0] + padded[3][2]*k[2][1] + padded[3][3]*k[2][2],
            padded[1][2]*k[0][0] + padded[1][3]*k[0][1] + padded[1][4]*k[0][2]
            + padded[2][2]*k[1][0] + padded[2][3]*k[1][1] + padded[2][4]*k[1][2]
            + padded[3][2]*k[2][0] + padded[3][3]*k[2][1] + padded[3][4]*k[2][2],
            padded[1][3]*k[0][0] + padded[1][4]*k[0][1] + padded[1][5]*k[0][2]
            + padded[2][3]*k[1][0] + padded[2][4]*k[1][1] + padded[2][5]*k[1][2]
            + padded[3][3]*k[2][0] + padded[3][4]*k[2][1] + padded[3][5]*k[2][2],
            padded[1][4]*k[0][0] + padded[1][5]*k[0][1] + padded[1][6]*k[0][2]
            + padded[2][4]*k[1][0] + padded[2][5]*k[1][1] + padded[2][6]*k[1][2]
            + padded[3][4]*k[2][0] + padded[3][5]*k[2][1] + padded[3][6]*k[2][2],
            padded[1][5]*k[0][0] + padded[1][6]*k[0][1] + padded[1][7]*k[0][2]
            + padded[2][5]*k[1][0] + padded[2][6]*k[1][1] + padded[2][7]*k[1][2]
            + padded[3][5]*k[2][0] + padded[3][6]*k[2][1] + padded[3][7]*k[2][2],
            padded[1][6]*k[0][0] + padded[1][7]*k[0][1] + padded[1][8]*k[0][2]
            + padded[2][6]*k[1][0] + padded[2][7]*k[1][1] + padded[2][8]*k[1][2]
            + padded[3][6]*k[2][0] + padded[3][7]*k[2][1] + padded[3][8]*k[2][2],
        ],
        [
            padded[2][0]*k[0][0] + padded[2][1]*k[0][1] + padded[2][2]*k[0][2]
            + padded[3][0]*k[1][0] + padded[3][1]*k[1][1] + padded[3][2]*k[1][2]
            + padded[4][0]*k[2][0] + padded[4][1]*k[2][1] + padded[4][2]*k[2][2],
            padded[2][1]*k[0][0] + padded[2][2]*k[0][1] + padded[2][3]*k[0][2]
            + padded[3][1]*k[1][0] + padded[3][2]*k[1][1] + padded[3][3]*k[1][2]
            + padded[4][1]*k[2][0] + padded[4][2]*k[2][1] + padded[4][3]*k[2][2],
            padded[2][2]*k[0][0] + padded[2][3]*k[0][1] + padded[2][4]*k[0][2]
            + padded[3][2]*k[1][0] + padded[3][3]*k[1][1] + padded[3][4]*k[1][2]
            + padded[4][2]*k[2][0] + padded[4][3]*k[2][1] + padded[4][4]*k[2][2],
            padded[2][3]*k[0][0] + padded[2][4]*k[0][1] + padded[2][5]*k[0][2]
            + padded[3][3]*k[1][0] + padded[3][4]*k[1][1] + padded[3][5]*k[1][2]
            + padded[4][3]*k[2][0] + padded[4][4]*k[2][1] + padded[4][5]*k[2][2],
            padded[2][4]*k[0][0] + padded[2][5]*k[0][1] + padded[2][6]*k[0][2]
            + padded[3][4]*k[1][0] + padded[3][5]*k[1][1] + padded[3][6]*k[1][2]
            + padded[4][4]*k[2][0] + padded[4][5]*k[2][1] + padded[4][6]*k[2][2],
            padded[2][5]*k[0][0] + padded[2][6]*k[0][1] + padded[2][7]*k[0][2]
            + padded[3][5]*k[1][0] + padded[3][6]*k[1][1] + padded[3][7]*k[1][2]
            + padded[4][5]*k[2][0] + padded[4][6]*k[2][1] + padded[4][7]*k[2][2],
            padded[2][6]*k[0][0] + padded[2][7]*k[0][1] + padded[2][8]*k[0][2]
            + padded[3][6]*k[1][0] + padded[3][7]*k[1][1] + padded[3][8]*k[1][2]
            + padded[4][6]*k[2][0] + padded[4][7]*k[2][1] + padded[4][8]*k[2][2],
        ],
        [
            padded[3][0]*k[0][0] + padded[3][1]*k[0][1] + padded[3][2]*k[0][2]
            + padded[4][0]*k[1][0] + padded[4][1]*k[1][1] + padded[4][2]*k[1][2]
            + padded[5][0]*k[2][0] + padded[5][1]*k[2][1] + padded[5][2]*k[2][2],
            padded[3][1]*k[0][0] + padded[3][2]*k[0][1] + padded[3][3]*k[0][2]
            + padded[4][1]*k[1][0] + padded[4][2]*k[1][1] + padded[4][3]*k[1][2]
            + padded[5][1]*k[2][0] + padded[5][2]*k[2][1] + padded[5][3]*k[2][2],
            padded[3][2]*k[0][0] + padded[3][3]*k[0][1] + padded[3][4]*k[0][2]
            + padded[4][2]*k[1][0] + padded[4][3]*k[1][1] + padded[4][4]*k[1][2]
            + padded[5][2]*k[2][0] + padded[5][3]*k[2][1] + padded[5][4]*k[2][2],
            padded[3][3]*k[0][0] + padded[3][4]*k[0][1] + padded[3][5]*k[0][2]
            + padded[4][3]*k[1][0] + padded[4][4]*k[1][1] + padded[4][5]*k[1][2]
            + padded[5][3]*k[2][0] + padded[5][4]*k[2][1] + padded[5][5]*k[2][2],
            padded[3][4]*k[0][0] + padded[3][5]*k[0][1] + padded[3][6]*k[0][2]
            + padded[4][4]*k[1][0] + padded[4][5]*k[1][1] + padded[4][6]*k[1][2]
            + padded[5][4]*k[2][0] + padded[5][5]*k[2][1] + padded[5][6]*k[2][2],
            padded[3][5]*k[0][0] + padded[3][6]*k[0][1] + padded[3][7]*k[0][2]
            + padded[4][5]*k[1][0] + padded[4][6]*k[1][1] + padded[4][7]*k[1][2]
            + padded[5][5]*k[2][0] + padded[5][6]*k[2][1] + padded[5][7]*k[2][2],
            padded[3][6]*k[0][0] + padded[3][7]*k[0][1] + padded[3][8]*k[0][2]
            + padded[4][6]*k[1][0] + padded[4][7]*k[1][1] + padded[4][8]*k[1][2]
            + padded[5][6]*k[2][0] + padded[5][7]*k[2][1] + padded[5][8]*k[2][2],
        ],
        [
            padded[4][0]*k[0][0] + padded[4][1]*k[0][1] + padded[4][2]*k[0][2]
            + padded[5][0]*k[1][0] + padded[5][1]*k[1][1] + padded[5][2]*k[1][2]
            + padded[6][0]*k[2][0] + padded[6][1]*k[2][1] + padded[6][2]*k[2][2],
            padded[4][1]*k[0][0] + padded[4][2]*k[0][1] + padded[4][3]*k[0][2]
            + padded[5][1]*k[1][0] + padded[5][2]*k[1][1] + padded[5][3]*k[1][2]
            + padded[6][1]*k[2][0] + padded[6][2]*k[2][1] + padded[6][3]*k[2][2],
            padded[4][2]*k[0][0] + padded[4][3]*k[0][1] + padded[4][4]*k[0][2]
            + padded[5][2]*k[1][0] + padded[5][3]*k[1][1] + padded[5][4]*k[1][2]
            + padded[6][2]*k[2][0] + padded[6][3]*k[2][1] + padded[6][4]*k[2][2],
            padded[4][3]*k[0][0] + padded[4][4]*k[0][1] + padded[4][5]*k[0][2]
            + padded[5][3]*k[1][0] + padded[5][4]*k[1][1] + padded[5][5]*k[1][2]
            + padded[6][3]*k[2][0] + padded[6][4]*k[2][1] + padded[6][5]*k[2][2],
            padded[4][4]*k[0][0] + padded[4][5]*k[0][1] + padded[4][6]*k[0][2]
            + padded[5][4]*k[1][0] + padded[5][5]*k[1][1] + padded[5][6]*k[1][2]
            + padded[6][4]*k[2][0] + padded[6][5]*k[2][1] + padded[6][6]*k[2][2],
            padded[4][5]*k[0][0] + padded[4][6]*k[0][1] + padded[4][7]*k[0][2]
            + padded[5][5]*k[1][0] + padded[5][6]*k[1][1] + padded[5][7]*k[1][2]
            + padded[6][5]*k[2][0] + padded[6][6]*k[2][1] + padded[6][7]*k[2][2],
            padded[4][6]*k[0][0] + padded[4][7]*k[0][1] + padded[4][8]*k[0][2]
            + padded[5][6]*k[1][0] + padded[5][7]*k[1][1] + padded[5][8]*k[1][2]
            + padded[6][6]*k[2][0] + padded[6][7]*k[2][1] + padded[6][8]*k[2][2],
        ],
        [
            padded[5][0]*k[0][0] + padded[5][1]*k[0][1] + padded[5][2]*k[0][2]
            + padded[6][0]*k[1][0] + padded[6][1]*k[1][1] + padded[6][2]*k[1][2]
            + padded[7][0]*k[2][0] + padded[7][1]*k[2][1] + padded[7][2]*k[2][2],
            padded[5][1]*k[0][0] + padded[5][2]*k[0][1] + padded[5][3]*k[0][2]
            + padded[6][1]*k[1][0] + padded[6][2]*k[1][1] + padded[6][3]*k[1][2]
            + padded[7][1]*k[2][0] + padded[7][2]*k[2][1] + padded[7][3]*k[2][2],
            padded[5][2]*k[0][0] + padded[5][3]*k[0][1] + padded[5][4]*k[0][2]
            + padded[6][2]*k[1][0] + padded[6][3]*k[1][1] + padded[6][4]*k[1][2]
            + padded[7][2]*k[2][0] + padded[7][3]*k[2][1] + padded[7][4]*k[2][2],
            padded[5][3]*k[0][0] + padded[5][4]*k[0][1] + padded[5][5]*k[0][2]
            + padded[6][3]*k[1][0] + padded[6][4]*k[1][1] + padded[6][5]*k[1][2]
            + padded[7][3]*k[2][0] + padded[7][4]*k[2][1] + padded[7][5]*k[2][2],
            padded[5][4]*k[0][0] + padded[5][5]*k[0][1] + padded[5][6]*k[0][2]
            + padded[6][4]*k[1][0] + padded[6][5]*k[1][1] + padded[6][6]*k[1][2]
            + padded[7][4]*k[2][0] + padded[7][5]*k[2][1] + padded[7][6]*k[2][2],
            padded[5][5]*k[0][0] + padded[5][6]*k[0][1] + padded[5][7]*k[0][2]
            + padded[6][5]*k[1][0] + padded[6][6]*k[1][1] + padded[6][7]*k[1][2]
            + padded[7][5]*k[2][0] + padded[7][6]*k[2][1] + padded[7][7]*k[2][2],
            padded[5][6]*k[0][0] + padded[5][7]*k[0][1] + padded[5][8]*k[0][2]
            + padded[6][6]*k[1][0] + padded[6][7]*k[1][1] + padded[6][8]*k[1][2]
            + padded[7][6]*k[2][0] + padded[2][7]*k[2][1] + padded[2][8]*k[2][2],
        ],
        [
            padded[6][0]*k[0][0] + padded[6][1]*k[0][1] + padded[6][2]*k[0][2]
            + padded[7][0]*k[1][0] + padded[7][1]*k[1][1] + padded[7][2]*k[1][2]
            + padded[8][0]*k[2][0] + padded[8][1]*k[2][1] + padded[8][2]*k[2][2],
            padded[6][1]*k[0][0] + padded[6][2]*k[0][1] + padded[6][3]*k[0][2]
            + padded[7][1]*k[1][0] + padded[7][2]*k[1][1] + padded[7][3]*k[1][2]
            + padded[8][1]*k[2][0] + padded[8][2]*k[2][1] + padded[8][3]*k[2][2],
            padded[6][2]*k[0][0] + padded[6][3]*k[0][1] + padded[6][4]*k[0][2]
            + padded[7][2]*k[1][0] + padded[7][3]*k[1][1] + padded[7][4]*k[1][2]
            + padded[8][2]*k[2][0] + padded[8][3]*k[2][1] + padded[8][4]*k[2][2],
            padded[6][3]*k[0][0] + padded[6][4]*k[0][1] + padded[6][5]*k[0][2]
            + padded[7][3]*k[1][0] + padded[7][4]*k[1][1] + padded[7][5]*k[1][2]
            + padded[8][3]*k[2][0] + padded[8][4]*k[2][1] + padded[8][5]*k[2][2],
            padded[6][4]*k[0][0] + padded[6][5]*k[0][1] + padded[6][6]*k[0][2]
            + padded[7][4]*k[1][0] + padded[7][5]*k[1][1] + padded[7][6]*k[1][2]
            + padded[8][4]*k[2][0] + padded[8][5]*k[2][1] + padded[8][6]*k[2][2],
            padded[6][5]*k[0][0] + padded[6][6]*k[0][1] + padded[6][7]*k[0][2]
            + padded[7][5]*k[1][0] + padded[7][6]*k[1][1] + padded[7][7]*k[1][2]
            + padded[8][5]*k[2][0] + padded[8][6]*k[2][1] + padded[8][7]*k[2][2],
            padded[6][6]*k[0][0] + padded[6][7]*k[0][1] + padded[6][8]*k[0][2]
            + padded[7][6]*k[1][0] + padded[7][7]*k[1][1] + padded[7][8]*k[1][2]
            + padded[8][6]*k[2][0] + padded[8][7]*k[2][1] + padded[8][8]*k[2][2],
        ],
    ];
    const result = Matrix.fullConvolve(src, k);

    return (
        expected[0][0] === result[0][0] &&
        expected[0][1] === result[0][1] &&
        expected[0][2] === result[0][2] &&
        expected[0][3] === result[0][3] &&
        expected[0][4] === result[0][4] &&
        expected[0][5] === result[0][5] &&
        expected[0][6] === result[0][6] &&
        expected[1][0] === result[1][0] &&
        expected[1][1] === result[1][1] &&
        expected[1][2] === result[1][2] &&
        expected[1][3] === result[1][3] &&
        expected[1][4] === result[1][4] &&
        expected[1][5] === result[1][5] &&
        expected[1][6] === result[1][6] &&
        expected[2][0] === result[2][0] &&
        expected[2][1] === result[2][1] &&
        expected[2][2] === result[2][2] &&
        expected[2][3] === result[2][3] &&
        expected[2][4] === result[2][4] &&
        expected[2][5] === result[2][5] &&
        expected[2][6] === result[2][6] &&
        expected[3][0] === result[3][0] &&
        expected[3][1] === result[3][1] &&
        expected[3][2] === result[3][2] &&
        expected[3][3] === result[3][3] &&
        expected[3][4] === result[3][4] &&
        expected[3][5] === result[3][5] &&
        expected[3][6] === result[3][6] &&
        expected[4][0] === result[4][0] &&
        expected[4][1] === result[4][1] &&
        expected[4][2] === result[4][2] &&
        expected[4][3] === result[4][3] &&
        expected[4][4] === result[4][4] &&
        expected[4][5] === result[4][5] &&
        expected[4][6] === result[4][6] &&
        expected[5][0] === result[5][0] &&
        expected[5][1] === result[5][1] &&
        expected[5][2] === result[5][2] &&
        expected[5][3] === result[5][3] &&
        expected[5][4] === result[5][4] &&
        expected[5][5] === result[5][5] &&
        expected[5][6] === result[5][6] &&
        expected[6][0] === result[6][0] &&
        expected[6][1] === result[6][1] &&
        expected[6][2] === result[6][2] &&
        expected[6][3] === result[6][3] &&
        expected[6][4] === result[6][4] &&
        expected[6][5] === result[6][5] &&
        expected[6][6] === result[6][6]
    );
});

console.log("");
console.log("Suite 3: Cost Functions");

console.log(`  Group 1: categorical cross-entropy`);
test("computes the cross-entropy loss", stats, () => {
    const prediction = new Float64Array([0.8, 0.5, 0.7]);
    const expected = new Float64Array([0, 1, 0]);
    let expectedLoss = 
        expected[0] * Math.log(prediction[0]) +
        expected[1] * Math.log(prediction[1]) +
        expected[2] * Math.log(prediction[2]);
    expectedLoss *= -1;
    const loss = Cost.categoricalCrossEntropy(prediction, expected);

    return expectedLoss === loss;
});
test("computes the derivative with softmax dL/dz", stats, () => {
    const prediction = new Float64Array([0.2, 0.8, 0.95, 0.11]);
    const expected = new Float64Array([1, 0, 0, 0]);
    const expectedDerivative = [
        prediction[0] - expected[0],
        prediction[1] - expected[1],
        prediction[2] - expected[2],
        prediction[3] - expected[3],
    ];
    const derivative = Cost.CCEDerivative_softmax(prediction, expected);
    
    return (
        expectedDerivative[0] === derivative[0] &&
        expectedDerivative[1] === derivative[1] &&
        expectedDerivative[2] === derivative[2] &&
        expectedDerivative[3] === derivative[3] &&
        expectedDerivative[4] === derivative[4]
    );
});

console.log(`  Group 2: binary cross-entropy`);
test("computes the cross-entropy loss", stats, () => {
    const prediction = 0;
    const label = 1;
    const EPSILON = 1e-9;
    const p = Math.max(prediction, EPSILON);
    const p2 = Math.max(1 - prediction, EPSILON);

    const expected = (label * Math.log(p)) + ((1 - label) * Math.log(p2));
    const result = Cost.binaryCrossEntropy(prediction, label);

    return expected === result;
});
test("computes the derivative with sigmoid dL/dz", stats, () => {
    const prediction = 0.6;
    const label = 0;
    
    const expected = Cost.BCEDerivative_sigmoid(prediction, label);
    const result = prediction - label;

    return expected === result;
});

console.log(`  Group 3: mean squared error`);
test("computes the mean squared error", stats, () => {
    const prediction = new Float64Array([0.8, 0.2, 0.6]);
    const label = new Float64Array([0, 1, 0]);
    let expected = Math.pow(label[0] - prediction[0], 2)
        + Math.pow(label[1] - prediction[1], 2)
        + Math.pow(label[2] - prediction[2], 2);
    expected /= prediction.length;
    const result = Cost.meanSquaredError(prediction, label);

    return expected === result;
});
test("computes the derivative dL/da", stats, () => {
    const prediction = new Float64Array([0.8, 0.2, 0.6]);
    const label = new Float64Array([0, 1, 0]);
    const expected = [
        (2 / prediction.length) * (prediction[0] - label[0]),
        (2 / prediction.length) * (prediction[1] - label[1]),
        (2 / prediction.length) * (prediction[2] - label[2]),
    ];
    const result = Cost.MSEDerivative(prediction, label);
    
    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});

console.log("");
console.log("Suite 4: Activation Functions");

console.log(`  Group 1: softmax`);
test("computes the normalized exponentiated values of a vector", stats, () => {
    const z = new Float64Array([12, 23, 9]);
    const sum = Math.exp(z[0]) + Math.exp(z[1]) + Math.exp(z[2]);
    const expected = [
        Math.exp(z[0]) / sum,
        Math.exp(z[1]) / sum,
        Math.exp(z[2]) / sum,
    ];
    const result = Activations.softmax(z);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});

console.log(`  Group 2: sigmoid`);
test("computes the smooth normalized values of a vector", stats, () => {
    const z = new Float64Array([6, 4, 9]);
    const expected = [
        1 / (1 + Math.exp(-z[0])),
        1 / (1 + Math.exp(-z[1])),
        1 / (1 + Math.exp(-z[2])),
    ];
    const result = Activations.sigmoid(z);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the sigmoid derivative", stats, () => {
    const z = new Float64Array([2, 4, 3]);
    const a = new Float64Array([
        1 / (1 + Math.exp(-z[0])),
        1 / (1 + Math.exp(-z[1])),
        1 / (1 + Math.exp(-z[2]))
    ]);
    const expected = [
        a[0] * (1 - a[0]),
        a[1] * (1 - a[1]),
        a[2] * (1 - a[2]),
    ];
    const result = Activations.sigmoidDerivative(a);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});

console.log(`  Group 3: swish`);
test("computes the smooth linear values of a vector", stats, () => {
    const z = new Float64Array([18, 23, 6]);
    const expected = [
        z[0] * (1 / (1 + Math.exp(-z[0]))),
        z[1] * (1 / (1 + Math.exp(-z[1]))),
        z[2] * (1 / (1 + Math.exp(-z[2]))),
    ];
    const result = Activations.swish(z);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the swish derivative", stats, () => {
    const z = new Float64Array([5, 12, 7]);
    const a = new Float64Array([
        z[0] * (1 / (1 + Math.exp(-z[0]))),
        z[1] * (1 / (1 + Math.exp(-z[1]))),
        z[2] * (1 / (1 + Math.exp(-z[2]))),
    ]);
    const expected = [
        a[0] + (1 / (1 + Math.exp(-z[0]))) * (1 - a[0]),
        a[1] + (1 / (1 + Math.exp(-z[1]))) * (1 - a[1]),
        a[2] + (1 / (1 + Math.exp(-z[2]))) * (1 - a[2]),
    ];
    const result = Activations.swishDerivative(a, z);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the smooth linear values of a tensor", stats, () => {
    const t = [
        [
            new Float64Array([2, 3]),
            new Float64Array([1, 6]),
        ],
        [
            new Float64Array([5, 8]),
            new Float64Array([5, 1]),
        ],
        [
            new Float64Array([4, 3]),
            new Float64Array([9, 2]),
        ],
    ];
    const expected = [
        [
            [t[0][0][0] * (1 / (1 + Math.exp(-t[0][0][0]))), t[0][0][1] * (1 / (1 + Math.exp(-t[0][0][1])))],
            [t[0][1][0] * (1 / (1 + Math.exp(-t[0][1][0]))), t[0][1][1] * (1 / (1 + Math.exp(-t[0][1][1])))],
        ],
        [
            [t[1][0][0] * (1 / (1 + Math.exp(-t[1][0][0]))), t[1][0][1] * (1 / (1 + Math.exp(-t[1][0][1])))],
            [t[1][1][0] * (1 / (1 + Math.exp(-t[1][1][0]))), t[1][1][1] * (1 / (1 + Math.exp(-t[1][1][1])))],
        ],
        [
            [t[2][0][0] * (1 / (1 + Math.exp(-t[2][0][0]))), t[2][0][1] * (1 / (1 + Math.exp(-t[2][0][1])))],
            [t[2][1][0] * (1 / (1 + Math.exp(-t[2][1][0]))), t[2][1][1] * (1 / (1 + Math.exp(-t[2][1][1])))],
        ],
    ];
    const result = Activations.swishT(t);
    
    return (
        expected[0][0][0] === result[0][0][0] &&
        expected[0][0][1] === result[0][0][1] &&
        expected[0][1][0] === result[0][1][0] &&
        expected[0][1][1] === result[0][1][1] &&
        expected[1][0][0] === result[1][0][0] &&
        expected[1][0][1] === result[1][0][1] &&
        expected[1][1][0] === result[1][1][0] &&
        expected[1][1][1] === result[1][1][1] &&
        expected[2][0][0] === result[2][0][0] &&
        expected[2][0][1] === result[2][0][1] &&
        expected[2][1][0] === result[2][1][0] &&
        expected[2][1][1] === result[2][1][1]
    );
});
test("computes the swish derivative of a tensor", stats, () => {
    const t = [
        [
            new Float64Array([2, 3]),
            new Float64Array([1, 6]),
        ],
        [
            new Float64Array([5, 8]),
            new Float64Array([5, 1]),
        ],
        [
            new Float64Array([4, 3]),
            new Float64Array([9, 2]),
        ],
    ];
    const a = [
        [
            [t[0][0][0] * (1 / (1 + Math.exp(-t[0][0][0]))), t[0][0][1] * (1 / (1 + Math.exp(-t[0][0][1])))],
            [t[0][1][0] * (1 / (1 + Math.exp(-t[0][1][0]))), t[0][1][1] * (1 / (1 + Math.exp(-t[0][1][1])))],
        ],
        [
            [t[1][0][0] * (1 / (1 + Math.exp(-t[1][0][0]))), t[1][0][1] * (1 / (1 + Math.exp(-t[1][0][1])))],
            [t[1][1][0] * (1 / (1 + Math.exp(-t[1][1][0]))), t[1][1][1] * (1 / (1 + Math.exp(-t[1][1][1])))],
        ],
        [
            [t[2][0][0] * (1 / (1 + Math.exp(-t[2][0][0]))), t[2][0][1] * (1 / (1 + Math.exp(-t[2][0][1])))],
            [t[2][1][0] * (1 / (1 + Math.exp(-t[2][1][0]))), t[2][1][1] * (1 / (1 + Math.exp(-t[2][1][1])))],
        ],
    ];
    const expected = [
        [
            [
                a[0][0][0] + (1 / (1 + Math.exp(-t[0][0][0]))) * (1 - a[0][0][0]),
                a[0][0][1] + (1 / (1 + Math.exp(-t[0][0][1]))) * (1 - a[0][0][1]),
            ],
            [
                a[0][1][0] + (1 / (1 + Math.exp(-t[0][1][0]))) * (1 - a[0][1][0]),
                a[0][1][1] + (1 / (1 + Math.exp(-t[0][1][1]))) * (1 - a[0][1][1]),
            ],
        ],
        [
            [
                a[1][0][0] + (1 / (1 + Math.exp(-t[1][0][0]))) * (1 - a[1][0][0]),
                a[1][0][1] + (1 / (1 + Math.exp(-t[1][0][1]))) * (1 - a[1][0][1]),
            ],
            [
                a[1][1][0] + (1 / (1 + Math.exp(-t[1][1][0]))) * (1 - a[1][1][0]),
                a[1][1][1] + (1 / (1 + Math.exp(-t[1][1][1]))) * (1 - a[1][1][1]),
            ],
        ],
        [
            [
                a[2][0][0] + (1 / (1 + Math.exp(-t[2][0][0]))) * (1 - a[2][0][0]),
                a[2][0][1] + (1 / (1 + Math.exp(-t[2][0][1]))) * (1 - a[2][0][1]),
            ],
            [
                a[2][1][0] + (1 / (1 + Math.exp(-t[2][1][0]))) * (1 - a[2][1][0]),
                a[2][1][1] + (1 / (1 + Math.exp(-t[2][1][1]))) * (1 - a[2][1][1]),
            ],
        ],
    ];
    const result = Activations.swishDerivativeT(a, t);
    
    return (
        expected[0][0][0] === result[0][0][0] &&
        expected[0][0][1] === result[0][0][1] &&
        expected[0][1][0] === result[0][1][0] &&
        expected[0][1][1] === result[0][1][1] &&
        expected[1][0][0] === result[1][0][0] &&
        expected[1][0][1] === result[1][0][1] &&
        expected[1][1][0] === result[1][1][0] &&
        expected[1][1][1] === result[1][1][1] &&
        expected[2][0][0] === result[2][0][0] &&
        expected[2][0][1] === result[2][0][1] &&
        expected[2][1][0] === result[2][1][0] &&
        expected[2][1][1] === result[2][1][1]
    );
});

console.log(`  Group 4: ReLU`);
test("computes the relu activations", stats, () => {
    const prediction = new Float64Array([0.5, -0.2, 0.8]);
    const expected = [
        prediction[0] > 0 ? prediction[0] : 0,
        prediction[1] > 0 ? prediction[1] : 0,
        prediction[2] > 0 ? prediction[2] : 0,
    ];
    const result = Activations.relu(prediction);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});
test("computes the relu derivative", stats, () => {
    const prediction = new Float64Array([12, -0.2, 2]);
    const expected = [
        prediction[0] > 0 ? 1 : 0,
        prediction[1] > 0 ? 1 : 0,
        prediction[2] > 0 ? 1 : 0,
    ];
    const result = Activations.reluDerivative(prediction);
    
    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2]
    );
});

console.log("");
console.log("Suite 5: E2E tests");
test("completes a forward pass", stats, () => {
    const W = Matrix.create(2, 2, -0.05, 0.05);
    const x = Vector.create(2, -0.05, 0.05);
    const b = Vector.create(2, -0.05, 0.05);
    const expected = [
        W[0][0]*x[0] + W[0][1]*x[1] + b[0],
        W[1][0]*x[0] + W[1][1]*x[1] + b[1]
    ];
    const result = NeuralNet.forwardPass(W, x, b);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1]
    );
});
test("completes a backward pass; computes the error signal", stats, () => {
    const W1 = Matrix.create(4, 2, -0.05, 0.05);
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Matrix.create(2, 4, -0.05, 0.05);
    const b2 = Vector.create(2, -0.05, 0.05);
    const input = Vector.create(2, -0.05, 0.05);
    const label = new Float64Array([0, 1]);

    const z1 = NeuralNet.forwardPass(W1, input, b1);
    const a1 = Activations.swish(z1);
    const z2 = NeuralNet.forwardPass(W2, a1, b2);
    const a2 = Activations.softmax(z2);

    const b2_grad = Cost.CCEDerivative_softmax(a2, label);
    const transposedW2 = [
        [W2[0][0], W2[1][0]],
        [W2[0][1], W2[1][1]],
        [W2[0][2], W2[1][2]],
        [W2[0][3], W2[1][3]],
    ];
    const expected = [
        (transposedW2[0][0]*b2_grad[0] + transposedW2[0][1]*b2_grad[1])
        * (a1[0] + (1 / (1 + Math.exp(-z1[0]))) * (1 - a1[0])),
        (transposedW2[1][0]*b2_grad[0] + transposedW2[1][1]*b2_grad[1])
        * (a1[1] + (1 / (1 + Math.exp(-z1[1]))) * (1 - a1[1])),
        (transposedW2[2][0]*b2_grad[0] + transposedW2[2][1]*b2_grad[1])
        * (a1[2] + (1 / (1 + Math.exp(-z1[2]))) * (1 - a1[2])),
        (transposedW2[3][0]*b2_grad[0] + transposedW2[3][1]*b2_grad[1])
        * (a1[3] + (1 / (1 + Math.exp(-z1[3]))) * (1 - a1[3])),
    ];
    const result = NeuralNet.backwardPass(
        W2, b2_grad, Activations.swishDerivative(a1, z1)
    );

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2] &&
        expected[3] === result[3]
    );
});
test("successfully trains a 2-layer XOR model", stats, () => {
    const inputs = [
        new Float64Array([0, 0]),
        new Float64Array([0, 1]),
        new Float64Array([1, 0]),
        new Float64Array([1, 1]),
    ]
    const labels = [
        new Float64Array([1, 0]),
        new Float64Array([0, 1]),
        new Float64Array([0, 1]),
        new Float64Array([1, 0]),
    ];
    const W1 = Matrix.create(2, 2, -0.05, 0.05);
    const b1 = Vector.create(2, -0.05, 0.05);
    const W2 = Matrix.create(2, 2, -0.05, 0.05);
    const b2 = Vector.create(2, -0.05, 0.05);
    const learningRate = 1;
    const EPOCH = 1000;

    for(let i = 0; i < EPOCH; i++) {
        for(let x = 0; x < inputs.length; x++) {
            const z1 = NeuralNet.forwardPass(W1, inputs[x], b1);
            const a1 = Activations.swish(z1);
            const z2 = NeuralNet.forwardPass(W2, a1, b2);
            const a2 = Activations.softmax(z2);

            const b2_grad = Cost.CCEDerivative_softmax(a2, labels[x]);
            const W2_grad = Vector.outer(b2_grad, a1);
            const b1_grad = NeuralNet.backwardPass(
                W2, b2_grad, Activations.swishDerivative(a1, z1)
            );
            const W1_grad = Vector.outer(b1_grad, inputs[x]);

            NeuralNet.gradientDescentMat(W2, W2_grad, learningRate);
            NeuralNet.gradientDescentVec(b2, b2_grad, learningRate);
            NeuralNet.gradientDescentMat(W1, W1_grad, learningRate);
            NeuralNet.gradientDescentVec(b1, b1_grad, learningRate);
        }
    }

    const results = new Array(4);
    for(let i = 0; i < inputs.length; i++) {
        const z1 = NeuralNet.forwardPass(W1, inputs[i], b1);
        const a1 = Activations.swish(z1);
        const z2 = NeuralNet.forwardPass(W2, a1, b2);
        const a2 = Activations.softmax(z2);

        results[i] = a2;
    }

    return (
        results[0].indexOf(Math.max(...results[0])) === labels[0].indexOf(1) &&
        results[1].indexOf(Math.max(...results[1])) === labels[1].indexOf(1) &&
        results[2].indexOf(Math.max(...results[2])) === labels[2].indexOf(1) &&
        results[3].indexOf(Math.max(...results[3])) === labels[3].indexOf(1)
    );
});
test("completes a convolution forward pass", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(2, 2, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(1, 1, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const expected = [
        [
            [
                mockRgbImage[0][0][0]*W1[0][0][0][0]
                + mockRgbImage[1][0][0]*W1[0][1][0][0]
                + mockRgbImage[2][0][0]*W1[0][2][0][0]
                + b1[0],
                mockRgbImage[0][0][1]*W1[0][0][0][0]
                + mockRgbImage[1][0][1]*W1[0][1][0][0]
                + mockRgbImage[2][0][1]*W1[0][2][0][0]
                + b1[0],
            ],
            [
                mockRgbImage[0][1][0]*W1[0][0][0][0]
                + mockRgbImage[1][1][0]*W1[0][1][0][0]
                + mockRgbImage[2][1][0]*W1[0][2][0][0]
                + b1[0],
                mockRgbImage[0][1][1]*W1[0][0][0][0]
                + mockRgbImage[1][1][1]*W1[0][1][0][0]
                + mockRgbImage[2][1][1]*W1[0][2][0][0]
                + b1[0],
            ],
        ],
        [
            [
                mockRgbImage[0][0][0]*W1[1][0][0][0]
                + mockRgbImage[1][0][0]*W1[1][1][0][0]
                + mockRgbImage[2][0][0]*W1[1][2][0][0]
                + b1[1],
                mockRgbImage[0][0][1]*W1[1][0][0][0]
                + mockRgbImage[1][0][1]*W1[1][1][0][0]
                + mockRgbImage[2][0][1]*W1[1][2][0][0]
                + b1[1],
            ],
            [
                mockRgbImage[0][1][0]*W1[1][0][0][0]
                + mockRgbImage[1][1][0]*W1[1][1][0][0]
                + mockRgbImage[2][1][0]*W1[1][2][0][0]
                + b1[1],
                mockRgbImage[0][1][1]*W1[1][0][0][0]
                + mockRgbImage[1][1][1]*W1[1][1][0][0]
                + mockRgbImage[2][1][1]*W1[1][2][0][0]
                + b1[1],
            ],
        ],
        [
            [
                mockRgbImage[0][0][0]*W1[2][0][0][0]
                + mockRgbImage[1][0][0]*W1[2][1][0][0]
                + mockRgbImage[2][0][0]*W1[2][2][0][0]
                + b1[2],
                mockRgbImage[0][0][1]*W1[2][0][0][0]
                + mockRgbImage[1][0][1]*W1[2][1][0][0]
                + mockRgbImage[2][0][1]*W1[2][2][0][0]
                + b1[2],
            ],
            [
                mockRgbImage[0][1][0]*W1[2][0][0][0]
                + mockRgbImage[1][1][0]*W1[2][1][0][0]
                + mockRgbImage[2][1][0]*W1[2][2][0][0]
                + b1[2],
                mockRgbImage[0][1][1]*W1[2][0][0][0]
                + mockRgbImage[1][1][1]*W1[2][1][0][0]
                + mockRgbImage[2][1][1]*W1[2][2][0][0]
                + b1[2],
            ],
        ],
        [
            [
                mockRgbImage[0][0][0]*W1[3][0][0][0]
                + mockRgbImage[1][0][0]*W1[3][1][0][0]
                + mockRgbImage[2][0][0]*W1[3][2][0][0]
                + b1[3],
                mockRgbImage[0][0][1]*W1[3][0][0][0]
                + mockRgbImage[1][0][1]*W1[3][1][0][0]
                + mockRgbImage[2][0][1]*W1[3][2][0][0]
                + b1[3],
            ],
            [
                mockRgbImage[0][1][0]*W1[3][0][0][0]
                + mockRgbImage[1][1][0]*W1[3][1][0][0]
                + mockRgbImage[2][1][0]*W1[3][2][0][0]
                + b1[3],
                mockRgbImage[0][1][1]*W1[3][0][0][0]
                + mockRgbImage[1][1][1]*W1[3][1][0][0]
                + mockRgbImage[2][1][1]*W1[3][2][0][0]
                + b1[3],
            ],
        ],
    ]
    const result = NeuralNet.convForward(mockRgbImage, W1, b1);

    return (
        expected[0][0][0] === result[0][0][0] &&
        expected[0][0][1] === result[0][0][1] &&
        expected[0][1][0] === result[0][1][0] &&
        expected[0][1][1] === result[0][1][1] &&
        expected[1][0][0] === result[1][0][0] &&
        expected[1][0][1] === result[1][0][1] &&
        expected[1][1][0] === result[1][1][0] &&
        expected[1][1][1] === result[1][1][1] &&
        expected[2][0][0] === result[2][0][0] &&
        expected[2][0][1] === result[2][0][1] &&
        expected[2][1][0] === result[2][1][0] &&
        expected[2][1][1] === result[2][1][1] &&
        expected[3][0][0] === result[3][0][0] &&
        expected[3][0][1] === result[3][0][1] &&
        expected[3][1][0] === result[3][1][0] &&
        expected[3][1][1] === result[3][1][1]
    );
});
test("successfully max-pools a conv layer output", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(2, 2, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(1, 1, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const z1 = NeuralNet.convForward(mockRgbImage, W1, b1);
    const expected = [
        [[Math.max(z1[0][0][0], z1[0][0][1], z1[0][1][0], z1[0][1][1])]],
        [[Math.max(z1[1][0][0], z1[1][0][1], z1[1][1][0], z1[1][1][1])]],
        [[Math.max(z1[2][0][0], z1[2][0][1], z1[2][1][0], z1[2][1][1])]],
        [[Math.max(z1[3][0][0], z1[3][0][1], z1[3][1][0], z1[3][1][1])]],
    ];
    const result = NeuralNet.pool(z1);

    return (
        expected[0][0][0] === result[0][0][0] &&
        expected[1][0][0] === result[1][0][0] &&
        expected[2][0][0] === result[2][0][0] &&
        expected[3][0][0] === result[3][0][0]
    );
});
test("completes a backward pass to a previous pooled layer", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(10, 10, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Array.from(
        { length: 5 },
        () => Array.from(
            { length: 4 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b2 = Vector.create(5, -0.05, 0.05);

    const z1 = NeuralNet.convForward(mockRgbImage, W1, b1);
    const z1_pooled = NeuralNet.pool(z1);
    const z2 = NeuralNet.convForward(z1_pooled, W2, b2);

    const dl_dz1_pooled = NeuralNet.convBackwardPrev(W2, z2);
    
    return (
        z1_pooled.length === dl_dz1_pooled.length &&
        z1_pooled[0].length === dl_dz1_pooled[0].length &&
        z1_pooled[0][0].length === dl_dz1_pooled[0][0].length
    );
});
test("completes an un-pooling backward pass", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(10, 10, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Array.from(
        { length: 5 },
        () => Array.from(
            { length: 4 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b2 = Vector.create(5, -0.05, 0.05);
    const mockActivations = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 8 },
            () => new Float64Array([1, 1, 1, 1, 1, 1, 1, 1])
        )
    );

    const z1 = NeuralNet.convForward(mockRgbImage, W1, b1);
    const z1_pooled = NeuralNet.pool(z1);
    const z2 = NeuralNet.convForward(z1_pooled, W2, b2);

    const dl_dz1_pooled = NeuralNet.convBackwardPrev(W2, z2);
    const dl_dz1 = NeuralNet.convBackward(z1, z1_pooled, dl_dz1_pooled, mockActivations);
    
    return (
        z1.length === dl_dz1.length &&
        z1[0].length === dl_dz1[0].length &&
        z1[0][0].length === dl_dz1[0][0].length
    );
});
test("successfully applies a global average pool to a conv layer output", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(10, 10, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Array.from(
        { length: 5 },
        () => Array.from(
            { length: 4 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b2 = Vector.create(5, -0.05, 0.05);

    const z1 = NeuralNet.convForward(mockRgbImage, W1, b1);
    const z1_pooled = NeuralNet.pool(z1);
    const z2 = NeuralNet.convForward(z1_pooled, W2, b2);
    const expected = [
        (z2[0][0][0] + z2[0][0][1] + z2[0][1][0] + z2[0][1][1]) / 4,
        (z2[1][0][0] + z2[1][0][1] + z2[1][1][0] + z2[1][1][1]) / 4,
        (z2[2][0][0] + z2[2][0][1] + z2[2][1][0] + z2[2][1][1]) / 4,
        (z2[3][0][0] + z2[3][0][1] + z2[3][1][0] + z2[3][1][1]) / 4,
        (z2[4][0][0] + z2[4][0][1] + z2[4][1][0] + z2[4][1][1]) / 4,
    ];
    const result = NeuralNet.globalAveragePool(z2);

    return (
        expected[0] === result[0] &&
        expected[1] === result[1] &&
        expected[2] === result[2] &&
        expected[3] === result[3] &&
        expected[4] === result[4]
    );
});
test("completes a backward pass from a dense layer to a conv layer", stats, () => {
    const mockRgbImage = Array.from(
        { length: 3 }, 
        () => Matrix.create(10, 10, -0.05, 0.05)
    );
    const W1 = Array.from(
        { length: 4 },
        () => Array.from(
            { length: 3 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Array.from(
        { length: 5 },
        () => Array.from(
            { length: 4 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b2 = Vector.create(5, -0.05, 0.05);
    const W3 = Matrix.create(3, 5, -0.05, 0.05);
    const b3 = Vector.create(3, -0.05, 0.05);
    const label = new Float64Array([1, 0, 0]);

    const z1 = NeuralNet.convForward(mockRgbImage, W1, b1);
    const a1 = Activations.swishT(z1);
    const a1_p = NeuralNet.pool(a1);
    const z2 = NeuralNet.convForward(a1_p, W2, b2);
    const a2 = Activations.swishT(z2);
    const a2_p = NeuralNet.globalAveragePool(a2);
    const z3 = NeuralNet.forwardPass(W3, a2_p, b3);
    const a3 = Activations.softmax(z3);

    const b3_grad = Cost.CCEDerivative_softmax(a3, label);
    const dl_dz2 = NeuralNet.convBackwardGAP(
        b3_grad, W3, 5, 2, 2,
        Activations.swishDerivativeT(a2, z2),
    );

    return (
        z2.length === dl_dz2.length &&
        z2[0].length === dl_dz2[0].length &&
        z2[0][0].length === dl_dz2[0][0].length
    );
});
test("successfully trains a 3-layer CNN to recognize logic gate images", stats, () => {
    const xorImage = [
        [
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
            new Float64Array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            new Float64Array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]
    ];
    const label = new Float64Array([1, 0, 0]);

    const W1 = Array.from(
        { length: 4 }, 
        () => [Matrix.create(3, 3, -0.05, 0.05)]
    );
    const b1 = Vector.create(4, -0.05, 0.05);
    const W2 = Array.from(
        { length: 8 },
        () => Array.from(
            { length: 4 },
            () => Matrix.create(3, 3, -0.05, 0.05)
        )
    );
    const b2 = Vector.create(8, -0.05, 0.05);
    const W3 = Matrix.create(3, 8, -0.05, 0.05);
    const b3 = Vector.create(3, -0.05, 0.05);
    const lr = 0.01;

    for(let i = 0; i < 1000; i++) {
        const z1 = NeuralNet.convForward(xorImage, W1, b1);
        const a1 = Activations.swishT(z1);
        const a1_p = NeuralNet.pool(a1);
        const z2 = NeuralNet.convForward(a1_p, W2, b2);
        const a2 = Activations.swishT(z2);
        const a2_p = NeuralNet.globalAveragePool(a2);
        const z3 = NeuralNet.forwardPass(W3, a2_p, b3);
        const a3 = Activations.softmax(z3);

        const b3_grad = Cost.CCEDerivative_softmax(a3, label);
        const W3_grad = Vector.outer(b3_grad, a2_p);
        const dl_dz2 = NeuralNet.convBackwardGAP(
            b3_grad, W3, 8, 3, 3,
            Activations.swishDerivativeT(a2, z2)
        );
        const W2_grad = NeuralNet.convKernelGrad(a1_p, dl_dz2);
        const b2_grad = NeuralNet.convBiasGrad(dl_dz2);
        const dl_da1_pool = NeuralNet.convBackwardPrev(W2, dl_dz2);
        const dl_dz1 = NeuralNet.convBackward(
            a1, a1_p, dl_da1_pool, 
            Activations.swishDerivativeT(a1, z1)
        );
        const W1_grad = NeuralNet.convKernelGrad(xorImage, dl_dz1);
        const b1_grad = NeuralNet.convBiasGrad(dl_dz1);

        NeuralNet.convGradientDescent(W1, W1_grad, lr);
        NeuralNet.convGradientDescent(W2, W2_grad, lr);
        NeuralNet.gradientDescentMat(W3, W3_grad, lr);
        NeuralNet.gradientDescentVec(b1, b1_grad, lr);
        NeuralNet.gradientDescentVec(b2, b2_grad, lr);
        NeuralNet.gradientDescentVec(b3, b3_grad, lr);
    }

    const z1 = NeuralNet.convForward(xorImage, W1, b1);
    const a1 = Activations.swishT(z1);
    const a1_p = NeuralNet.pool(a1);
    const z2 = NeuralNet.convForward(a1_p, W2, b2);
    const a2 = Activations.swishT(z2);
    const a2_p = NeuralNet.globalAveragePool(a2);
    const z3 = NeuralNet.forwardPass(W3, a2_p, b3);
    const a3 = Activations.softmax(z3);

    return a3.indexOf(Math.max(...a3)) === label.indexOf(1);
});

console.log("");
summary(stats);
