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

console.log("");
summary(stats);
