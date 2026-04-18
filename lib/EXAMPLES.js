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
    process.stdout.write(`Info:\nthis is a mock image classification model.\nIt takes in a 12x12 grayscale image containing a "logo" for a specific logic gate (XOR, AND, OR).\nThe label corresponds to the image's logo;\n[1, 0, 0] means the image is of a XOR logo, and so on.\n`);
};

const Transformer = () => {
    const text = [
        "truly interesting endeavor for a film",
        "meh",
        "hate this movie",
        "this is an extremely fantastic film",
        "it ' s alright",
        "terrible film",
    ];
    const labels = [
        new Float64Array([0, 0, 1]),
        new Float64Array([0, 1, 0]),
        new Float64Array([1, 0, 0]),
        new Float64Array([0, 0, 1]),
        new Float64Array([0, 1, 0]),
        new Float64Array([1, 0, 0]),
    ];
    const vocabulary = {
        "truly": 0,
        "interesting": 1,
        "endeavor": 2,
        "for": 3,
        "a": 4,
        "film": 5,
        "meh": 6,
        "hate": 7,
        "this": 8,
        "movie": 9,
        "is": 10,
        "an": 11,
        "extremely": 12,
        "fantastic": 13,
        "it": 14,
        "'": 15,
        "s": 16,
        "alright": 17,
        "terrible": 18,
        "<CLS>": 19,
    };
    const d_model = 64;
    const max_seq_len = 100;
    const n_class = 3;
    const vocab_size = 20;
    const POS = NeuralNet.initSinusoidal(max_seq_len, d_model);
    const E = Matrix.create(vocab_size, d_model, -0.05, 0.05);
    const Wq = Matrix.create(d_model, d_model, -0.05, 0.05);
    const Wk = Matrix.create(d_model, d_model, -0.05, 0.05);
    const Wv = Matrix.create(d_model, d_model, -0.05, 0.05);
    const scale1 = Vector.create(d_model, -0.05, 0.05);
    const shift1 = Vector.create(d_model, -0.05, 0.05);
    const W1 = Matrix.create(d_model, d_model * 4, -0.05, 0.05);
    const b1 = Vector.create(d_model * 4, -0.05, 0.05);
    const W2 = Matrix.create(d_model * 4, d_model, -0.05, 0.05);
    const b2 = Vector.create(d_model, -0.05, 0.05);
    const scale2 = Vector.create(d_model, -0.05, 0.05);
    const shift2 = Vector.create(d_model, -0.05, 0.05);
    const Wc = Matrix.create(n_class, d_model, -0.05, 0.05);
    const bc = Vector.create(n_class, -0.05, 0.05);
    const lr = 5e-2;
    const EPOCHS = 1000;

    let it = 0;
    for(let i = 0; i < EPOCHS; i++) {
        for(let x = 0; x < text.length; x++) {
            it++;

            const embeddings = NeuralNet.tokenize(text[x], vocabulary, E, POS);
            const [ 
                attention_out,
                attention_cache
            ] = NeuralNet.attention(embeddings, Wq, Wk, Wv, d_model);
            const attention_residual = Matrix.add(attention_out, embeddings);
            const [
                attention_norm_out,
                attention_norm_cache
            ] = NeuralNet.layerNorm(attention_residual, scale1, shift1);
            const FFN_z1 = NeuralNet.transformerFFN(attention_norm_out, W1, b1);
            const FFN_a1 = FFN_z1.map(vector => Activations.swish(vector));
            const FFN_z2 = NeuralNet.transformerFFN(FFN_a1, W2, b2);
            const FFN_residual = Matrix.add(FFN_z2, attention_norm_out);
            const [
                FFN_norm_out,
                FFN_norm_cache,
            ] = NeuralNet.layerNorm(FFN_residual, scale2, shift2);
            const CLS = FFN_norm_out[0];
            const z = NeuralNet.forwardPass(Wc, CLS, bc);
            const a = Activations.softmax(z);

            const dL_dz = Cost.CCEDerivative_softmax(a, labels[x]);
            const [
                bc_grad,
                Wc_grad,
                dL_dh2,
            ] = NeuralNet.classificationHeadBackward(dL_dz, CLS, Wc);
            const [
                scale2_grad, 
                shift2_grad, 
                dL_dr2
            ] = NeuralNet.LNbackward(
                dL_dh2, 
                FFN_norm_cache.CLS_normalized, 
                FFN_norm_cache.var, 
                scale2,
                d_model,
            );
            const [
                b2_grad,
                W2_grad,
                b1_grad,
                W1_grad,
                dL_dxFFN
            ] = NeuralNet.FFNbackward(
                dL_dr2,
                FFN_a1[0],
                FFN_z1[0],
                attention_norm_out[0],
                W1,
                W2
            );
            const dL_dh1 = Vector.add(dL_dr2, dL_dxFFN);
            const [
                scale1_grad,
                shift1_grad,
                dL_dr1_v
            ] = NeuralNet.LNbackward(
                dL_dh1,
                attention_norm_cache.CLS_normalized,
                attention_norm_cache.var,
                scale1,
                d_model
            );
            const [
                Wq_grad,
                Wk_grad,
                Wv_grad,
                dL_dxAttention,
                dL_dr1
            ] = NeuralNet.attentionBackward(
                embeddings,
                dL_dr1_v,
                attention_cache.A,
                attention_cache.Q,
                attention_cache.K,
                attention_cache.V,
                Wq,
                Wk,
                Wv,
                d_model,
            );
            const dL_dx = Matrix.add(dL_dr1, dL_dxAttention);
            const E_grad = NeuralNet.embeddingBackward(dL_dx, text[x], E, d_model, vocabulary);

            NeuralNet.gradientDescentMat(E, E_grad, lr);
            NeuralNet.gradientDescentMat(Wq, Wq_grad, lr);
            NeuralNet.gradientDescentMat(Wk, Wk_grad, lr);
            NeuralNet.gradientDescentMat(Wv, Wv_grad, lr);
            NeuralNet.gradientDescentVec(scale1, scale1_grad, lr);
            NeuralNet.gradientDescentVec(shift1, shift1_grad, lr);
            NeuralNet.gradientDescentMat(W1, W1_grad, lr);
            NeuralNet.gradientDescentVec(b1, b1_grad, lr);
            NeuralNet.gradientDescentMat(W2, W2_grad, lr);
            NeuralNet.gradientDescentVec(b2, b2_grad, lr);
            NeuralNet.gradientDescentVec(scale2, scale2_grad, lr);
            NeuralNet.gradientDescentVec(shift2, shift2_grad, lr);
            NeuralNet.gradientDescentMat(Wc, Wc_grad, lr);
            NeuralNet.gradientDescentVec(bc, bc_grad, lr);

            process.stdout.write(`\rTraining Transformer: iteration ${it}`);
        }
    }
    process.stdout.write(`\nTransformer Training complete.`);

    const results = new Array(6);
    for(let i = 0; i < text.length; i++) {
        const embeddings = NeuralNet.tokenize(text[i], vocabulary, E, POS);
        const [ 
            attention_out,
            attention_cache
        ] = NeuralNet.attention(embeddings, Wq, Wk, Wv, d_model);
        const attention_residual = Matrix.add(attention_out, embeddings);
        const [
            attention_norm_out,
            attention_norm_cache
        ] = NeuralNet.layerNorm(attention_residual, scale1, shift1);
        const FFN_z1 = NeuralNet.transformerFFN(attention_norm_out, W1, b1);
        const FFN_a1 = FFN_z1.map(vector => Activations.swish(vector));
        const FFN_z2 = NeuralNet.transformerFFN(FFN_a1, W2, b2);
        const FFN_residual = Matrix.add(FFN_z2, attention_norm_out);
        const [
            FFN_norm_out,
            FFN_norm_cache,
        ] = NeuralNet.layerNorm(FFN_residual, scale2, shift2);
        const CLS = FFN_norm_out[0];
        const z = NeuralNet.forwardPass(Wc, CLS, bc);
        const a = Activations.softmax(z);

        results[i] = a;
    }

    process.stdout.write('\n\n');
    process.stdout.write('Results:\n');
    for(let i = 0; i < 6; i++) {
        process.stdout.write(`Label: [${labels[i].join(", ")}], `);
        process.stdout.write(`Predicted: [${
            results[i].map(num => num.toFixed(4)).join(", ")
        }]\n`);
    }
    process.stdout.write('\n');
    process.stdout.write(`Info:\nThis is a sentiment analysis model.\nIt takes in a rating of a movie and analyzes whether it is negative, neutral, or positive.\nThe label corresponds to the positivity/negativity of the rating;\nA label of [1, 0, 0] means negative, and so on.\n`);
}

MLP();
process.stdout.write(`\n-------------------------------------------------------------\n\n`);
CNN();
process.stdout.write(`\n-------------------------------------------------------------\n\n`);
Transformer();
