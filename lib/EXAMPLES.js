import { Matrix, Vector, Activations, Cost, NeuralNet } from "./index.js";

const MLP = () => {
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

    for(let i = 0; i < 1000; i++) {
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
        process.stdout.write(`\rTraining MLP: iteration ${i + 1}`);
    }
    process.stdout.write(`\nMLP Training complete.`);

    const results = new Array(4);
    for(let i = 0; i < inputs.length; i++) {
        const z1 = NeuralNet.forwardPass(W1, inputs[i], b1);
        const a1 = Activations.swish(z1);
        const z2 = NeuralNet.forwardPass(W2, a1, b2);
        const a2 = Activations.softmax(z2);

        results[i] = a2;
    }

    process.stdout.write('\n\n');
    process.stdout.write('Results:\n');
    for(let i = 0; i < 4; i++) {
        process.stdout.write(`Input: [${inputs[i].join(", ")}], `);
        process.stdout.write(`Label: [${labels[i].join(", ")}], `);
        process.stdout.write(`Predicted: [${
            results[i].map(num => num.toFixed(4)).join(", ")
        }]\n`);
    }
    process.stdout.write('\n');
    process.stdout.write(`Info:\nthis is a XOR classification model.\nIt takes in a tuple of two numbers that represents the state of two bits.\nThe 1's index at the label corresponds to the binary output;\n[1, 0] means 0 or FALSE, and [0, 1] means 1 or TRUE.\n`);
}

const CNN = () => {
    // Each individual image example is intentionally nested twice for multi-channel support; the forward pass function always requires a 3D tensor.
    const images = [
        [
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
        ],
        [
            [
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]),
                new Float64Array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]),
                new Float64Array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]),
                new Float64Array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        ],
        [
            [
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]),
                new Float64Array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]),
                new Float64Array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
                new Float64Array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]),
                new Float64Array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                new Float64Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        ],
    ]
    const labels = [
        new Float64Array([1, 0, 0]),
        new Float64Array([0, 1, 0]),
        new Float64Array([0, 0, 1]),
    ];

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
    const lr = 0.05;

    for(let i = 0; i < 5000; i++) {
        for(let x = 0; x < images.length; x++) {
            const z1 = NeuralNet.convForward(images[x], W1, b1);
            const a1 = Activations.swishT(z1);
            const a1_p = NeuralNet.pool(a1);
            const z2 = NeuralNet.convForward(a1_p, W2, b2);
            const a2 = Activations.swishT(z2);
            const a2_p = NeuralNet.globalAveragePool(a2);
            const z3 = NeuralNet.forwardPass(W3, a2_p, b3);
            const a3 = Activations.softmax(z3);

            const b3_grad = Cost.CCEDerivative_softmax(a3, labels[x]);
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
            const W1_grad = NeuralNet.convKernelGrad(images[x], dl_dz1);
            const b1_grad = NeuralNet.convBiasGrad(dl_dz1);

            NeuralNet.convGradientDescent(W1, W1_grad, lr);
            NeuralNet.convGradientDescent(W2, W2_grad, lr);
            NeuralNet.gradientDescentMat(W3, W3_grad, lr);
            NeuralNet.gradientDescentVec(b1, b1_grad, lr);
            NeuralNet.gradientDescentVec(b2, b2_grad, lr);
            NeuralNet.gradientDescentVec(b3, b3_grad, lr);
        }
        process.stdout.write(`\rTraining CNN: iteration ${i + 1}`);
    }
    process.stdout.write(`\nCNN Training complete.`);

    const results = new Array(3);
    for(let i = 0; i < images.length; i++) {
        const z1 = NeuralNet.convForward(images[i], W1, b1);
        const a1 = Activations.swishT(z1);
        const a1_p = NeuralNet.pool(a1);
        const z2 = NeuralNet.convForward(a1_p, W2, b2);
        const a2 = Activations.swishT(z2);
        const a2_p = NeuralNet.globalAveragePool(a2);
        const z3 = NeuralNet.forwardPass(W3, a2_p, b3);
        const a3 = Activations.softmax(z3);

        results[i] = a3;
    }

    process.stdout.write('\n\n');
    process.stdout.write('Results:\n');
    for(let i = 0; i < 3; i++) {
        process.stdout.write(`Label: [${labels[i].join(", ")}], `);
        process.stdout.write(`Predicted: [${
            results[i].map(num => num.toFixed(4)).join(", ")
        }]\n`);
    }
    process.stdout.write('\n');
    process.stdout.write(`Info:\nthis is a mock image classification model.\nIt takes in a 12x12 grayscale image containing a "logo" for a specific logic gate (XOR, AND, OR).\nThe label corresponds to what logo the image has;\n[1, 0, 0] means the image is of a XOR logo, and so on.\n`);
};

MLP();
process.stdout.write(`\n-------------------------------------------------------------\n\n`);
CNN();
