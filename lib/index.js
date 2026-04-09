import { isArrayBufferView } from "node:util/types";

export class Shared {
    static validateStructure(a, b) {
        if(!isArrayBufferView(a) || !isArrayBufferView(b)) 
            throw new TypeError("Both vectors must be a Float64Array");
        if(!a.length || !b.length) 
            throw new TypeError("Both vectors must have positive length");

        return true;
    }
    static validateStructureSingle(a) {
        if(!isArrayBufferView(a))
            throw new TypeError("Vector must be a Float64Array");
        if(a.length <= 0)
            throw new TypeError("Vector must have positive length");

        return true;
    }
    static validateSameLength(a, b) {
        if(a.length !== b.length) 
            throw new TypeError("Both vectors must have the same length for this operation");
        
        return true;
    }
    static validateStructureMat(a, b) {
        if(
            !Array.isArray(a) || !Array.isArray(b) ||
            !isArrayBufferView(a[0]) || !isArrayBufferView(b[0])
        ) throw new TypeError("Both matrices must be a 2D Float64Array");
        if(
            !a.length || !b.length ||
            !a[0].length || !b[0].length
        ) throw new TypeError("Both matrices must have positive row and column length");

        return true;
    }
    static validateStructureMatSingle(a) {
        if(!Array.isArray(a) || !isArrayBufferView(a[0]))
            throw new TypeError("Matrix must be a 2D Float64Array");
        if(!a.length || !a[0].length)
            throw new TypeError("Matrix must have positive row and column length");

        return true;
    }
    static validateSameShape(a, b) {
        if(
            a.length !== b.length ||
            a[0].length !== b[0].length
        ) throw new TypeError("Both matrices must have the same dimensions for this operation");

        return true;
    }
}

export class Vector {
    static create(length, min, max) {
        if(length <= 0 || typeof length !== "number")
            throw new Error("Length parameter must be a positive number");
        if(typeof min !== "number" || typeof max !== "number")
            throw new Error("Both range limits must be a number");

        const vec = new Float64Array(length);
        for(let i = 0; i < length; i++) {
            vec[i] = Random.distr(min, max);
        }

        return vec;
    }
    static add(a, b) {
        Shared.validateStructure(a, b);
        Shared.validateSameLength(a, b);

        const vec = new Float64Array(a.length);
        for(let i = 0; i < a.length; i++) {
            vec[i] = a[i] + b[i];
        }

        return vec;
    }
    static subtract(a, b) {
        Shared.validateStructure(a, b);
        Shared.validateSameLength(a, b);

        const vec = new Float64Array(a.length);
        for(let i = 0; i < a.length; i++) {
            vec[i] = a[i] - b[i];
        }

        return vec;
    }
    static dot(a, b) {
        Shared.validateStructure(a, b);
        Shared.validateSameLength(a, b);

        let sum = 0;
        for(let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }
    static elementwise(a, b) {
        Shared.validateStructure(a, b);
        Shared.validateSameLength(a, b);

        const vec = new Float64Array(a.length);
        for(let i = 0; i < a.length; i++) {
            vec[i] = a[i] * b[i];
        }
        
        return vec;
    }
    static outer(a, b) {
        Shared.validateStructure(a, b);

        const mat = new Array(a.length);
        for(let i = 0; i < a.length; i++) {
            mat[i] = new Float64Array(b.length);
            for(let j = 0; j < b.length; j++) {
                mat[i][j] = a[i] * b[j];
            }
        }
        
        return mat;
    }
}

export class Matrix {
    static create(rows, cols, min, max) {
        if(
            rows <= 0 || typeof rows !== "number" ||
            cols <= 0 || typeof cols !== "number"
        ) throw new Error("Both parameters must be a positive number");
        if(typeof min !== "number" || typeof max !== "number")
            throw new Error("Both range limits must be a number");

        const mat = new Array(rows);
        for(let i = 0; i < rows; i++) {
            mat[i] = new Float64Array(cols);
            for(let j = 0; j < cols; j++) {
                mat[i][j] = Random.distr(min, max);
            }
        }

        return mat;
    }
    static add(a, b) {
        Shared.validateStructureMat(a, b);
        Shared.validateSameShape(a, b);

        const mat = new Array(a.length);
        for(let i = 0; i < a.length; i++) {
            mat[i] = new Float64Array(a[0].length);
            for(let j = 0; j < a[0].length; j++) {
                mat[i][j] = a[i][j] + b[i][j];
            }
        }

        return mat;
    }
    static subtract(a, b) {
        Shared.validateStructureMat(a, b);
        Shared.validateSameShape(a, b);

        const mat = new Array(a.length);
        for(let i = 0; i < a.length; i++) {
            mat[i] = new Float64Array(a[0].length);
            for(let j = 0; j < a[0].length; j++) {
                mat[i][j] = a[i][j] - b[i][j];
            }
        }

        return mat;
    }
    static multiply(a, b) {
        Shared.validateStructureMat(a, b);
        if(a[0].length !== b.length) 
            throw new TypeError("The columns of A must be equal to the rows of B\nfor matrix multiplication.\nTry transposing a matrix or flipping the order of arguments");

        const rowsA = a.length;
        const colsA = a[0].length;
        const colsB = b[0].length;
        const mat = new Array(a.length);
        for(let i = 0; i < rowsA; i++) {
            mat[i] = new Float64Array(b[0].length);
            for(let k = 0; k < colsA; k++) {
                for(let j = 0; j < colsB; j++) {
                    mat[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return mat;
    }
    static elementwise(a, b) {
        Shared.validateStructureMat(a, b);
        Shared.validateSameShape(a, b);

        const mat = new Array(a.length);
        for(let i = 0; i < a.length; i++) {
            mat[i] = new Float64Array(a[0].length);
            for(let j = 0; j < a[0].length; j++) {
                mat[i][j] = a[i][j] * b[i][j];
            }
        }

        return mat;
    }
    static matrixVector(m, v) {
        Shared.validateStructureSingle(v);
        Shared.validateStructureMatSingle(m);
        if(v.length !== m[0].length)
            throw new TypeError("Vector length must be the same as matrix column length\nfor matrix vector multiplication");

        const vec = new Float64Array(m.length);
        for(let i = 0; i < m.length; i++) {
            vec[i] = Vector.dot(v, m[i]);
        }

        return vec;
    }

    static addScalar(m, s) {
        Shared.validateStructureMatSingle(m);

        const mat = new Array(m.length);
        for(let i = 0; i < m.length; i++) {
            mat[i] = new Float64Array(m[0].length);
            for(let j = 0; j < m[0].length; j++) {
                mat[i][j] = m[i][j] + s;
            }
        }

        return mat;
    }

    static convolve(src, kernel) {
        Shared.validateStructureMatSingle(src);
        Shared.validateStructureMatSingle(kernel);
        if(kernel.length > src.length || kernel[0].length > src[0].length)
            throw new TypeError("Kernel cannot have larger size than source matrix");

        const outRows = src.length - kernel.length + 1;
        const outCols = src[0].length - kernel[0].length + 1;
        const mat = Array.from({ length: outRows }, () => new Float64Array(outCols));

        for(let i = 0; i < outRows; i++) {
            for(let j = 0; j < outCols; j++) {
                let sum = 0;
                for(let k = 0; k < kernel.length; k++) {
                    for(let l = 0; l < kernel[0].length; l++) {
                        sum += src[i + k][j + l] * kernel[k][l];
                    }
                }
                mat[i][j] = sum;
            }
        }

        return mat;
    }

    static transpose(m) {
        Shared.validateStructureMatSingle(m);

        const mt = Array.from({ length: m[0].length }, () => new Float64Array(m.length));
        for(let i = 0; i < m.length; i++) {
            for(let j = 0; j < m[0].length; j++) {
                mt[j][i] = m[i][j];
            }
        }

        return mt;
    }
    static flip(m) {
        Shared.validateStructureMatSingle(m);

        const flipped = Array.from({ length: m.length }, () => new Float64Array(m[0].length));
        const rows = m.length - 1;
        const cols = m[0].length - 1;
        for(let i = 0; i < m.length; i++) {
            for(let j = 0; j < m[0].length; j++) {
                flipped[i][j] = m[rows - i][cols - j];
            }
        }

        return flipped;
    }
    static reshape(v, rows, cols) {
        Shared.validateStructureSingle(v);

        const m = Array.from({ length: rows }, () => new Float64Array(cols));
        for(let i = 0; i < v.length; i++) {
            m[Math.floor(i / cols)][i % cols] = v[i];
        }

        return m;
    }
    static flatten(m) {
        Shared.validateStructureMatSingle(m);

        const v = new Float64Array(m.length * m[0].length);
        for(let i = 0; i < m.length; i++) {
            for(let j = 0; j < m[0].length; j++) {
                v[i * m[0].length + j] = m[i][j];
            }
        }

        return v;
    }
}

export class Random {
    static seed = Date.now();

    static num() {
        // derived from "Xorshift RNGs" by George Marsaglia (2003)
        this.seed ^= this.seed << 13;
        this.seed ^= this.seed >> 17;
        this.seed ^= this.seed << 5;
        // convert seed to unsigned 32-bit int
        const ui32 = this.seed >>> 0;
        // max of 32-bit int, plus 1 to get interval of [0, 1)
        const MAXp1 = 4294967296;

        return ui32 / MAXp1;
    }
    static distr(min, max) {
        const rand = this.num();
        
        return Number(
            (rand * (max - min) + min).toFixed(4)
        );
    }
}

export class Cost {
    static EPSILON = 1e-9

    static categoricalCrossEntropy(prediction, expected) {
        Shared.validateStructure(prediction, expected);
        Shared.validateSameLength(prediction, expected);

        let loss = 0;
        for(let i = 0; i < prediction.length; i++) {
            const p = Math.max(prediction[i], this.EPSILON);
            loss += expected[i] * Math.log(p);
        }
        loss *= -1;

        return loss;
    }
    static CCEDerivative_softmax(prediction, expected) {
        Shared.validateStructure(prediction, expected);
        Shared.validateSameLength(prediction, expected);

        const dL_dz = new Float64Array(prediction.length);
        for(let i = 0; i < prediction.length; i++) {
            dL_dz[i] = prediction[i] - expected[i];
        }

        return dL_dz;
    }

    static binaryCrossEntropy(prediction, expected) {
        const p = Math.max(prediction, this.EPSILON);
        const p2 = Math.max(1 - prediction, this.EPSILON);

        return (expected * Math.log(p)) + ((1 - expected) * (Math.log(p2)));
    }
    static BCEDerivative_sigmoid(prediction, expected) {
        return prediction - expected;
    }

    static meanSquaredError(prediction, expected) {
        Shared.validateStructure(prediction, expected);
        Shared.validateSameLength(prediction, expected);

        let loss = 0;
        for(let i = 0; i < prediction.length; i++) {
            loss += Math.pow(expected[i] - prediction[i], 2);
        }

        return loss / prediction.length;
    }
    static MSEDerivative(prediction, expected) {
        Shared.validateStructure(prediction, expected);
        Shared.validateSameLength(prediction, expected);

        const lossVec = new Float64Array(prediction.length);
        for(let i = 0; i < prediction.length; i++) {
            lossVec[i] = (2 / prediction.length) * (prediction[i] - expected[i]);
        }

        return lossVec;
    }
}

export class Activations {
    static softmax(z) {
        Shared.validateStructureSingle(z);

        const a = new Float64Array(z.length);
        let sum = 0;
        for(let i = 0; i < z.length; i++) {
            a[i] = Math.exp(z[i]);
            sum += a[i];
        }
        for(let i = 0; i < z.length; i++) {
            a[i] /= sum;
        }

        return a;
    }
    static sigmoid(z) {
        Shared.validateStructureSingle(z);

        const a = new Float64Array(z.length);
        for(let i = 0; i < z.length; i++) {
            a[i] = 1 / (1 + Math.exp(-z[i]));
        }

        return a;
    }
    static sigmoidDerivative(a) {
        Shared.validateStructureSingle(a);

        const da_dz = new Float64Array(a.length);
        for(let i = 0; i < a.length; i++) {
            da_dz[i] = a[i] * (1 - a[i]);
        }

        return da_dz;
    }
    static swish(z) {
        Shared.validateStructureSingle(z);
        
        const a = new Float64Array(z.length);
        for(let i = 0; i < z.length; i++) {
            a[i] = z[i] * (1 / (1 + Math.exp(-z[i])));
        }

        return a;
    }
    static swishT(z) {
        const a = Array.from({ length: z.length }, () => new Array(z[0].length));
        for(let i = 0; i < z.length; i++) {
            for(let j = 0; j < z[0].length; j++) {
                const va = new Float64Array(z[i][j].length);
                for(let k = 0; k < z[i][j].length; k++) {
                    va[k] = z[i][j][k] * (1 / (1 + Math.exp(-z[i][j][k])));
                }
                a[i][j] = va;
            }
        }

        return a;
    }
    static swishDerivativeT(a, z) {
        const da_dz = Array.from({ length: z.length }, () => new Array(z[0].length));
        for(let i = 0; i < z.length; i++) {
            for(let j = 0; j < z[0].length; j++) {
                const vda_dz = new Float64Array(a[i][j].length);
                for(let k = 0; k < a[i][j].length; k++) {
                    vda_dz[k] = a[i][j][k] + (1 / (1 + Math.exp(-z[i][j][k]))) * (1 - a[i][j][k]);
                }
                da_dz[i][j] = vda_dz;
            }
        }

        return da_dz;
    }
    static swishDerivative(a, z) {
        Shared.validateStructure(a, z);
        Shared.validateSameLength(a, z);

        const da_dz = new Float64Array(a.length);
        for(let i = 0; i < a.length; i++) {
            da_dz[i] = a[i] + (1 / (1 + Math.exp(-z[i]))) * (1 - a[i]);
        }

        return da_dz;
    }
    static relu(z) {
        Shared.validateStructureSingle(z);

        const a = new Float64Array(z.length);
        for(let i = 0; i < z.length; i++) {
            a[i] = z[i] > 0 ? z[i] : 0;
        }

        return a;
    }
    static reluDerivative(z) {
        Shared.validateStructureSingle(z);

        const da_dz = new Float64Array(z.length);
        for(let i = 0; i < z.length; i++) {
            da_dz[i] = z[i] > 0 ? 1 : 0;
        }

        return da_dz;
    }
}

export class NeuralNet {
    static forwardPass(W, x, b) {
        const Wx = Matrix.matrixVector(W, x);
        return Vector.add(Wx, b);
    }
    static convForwardSingleChannel(src, kernels, b) {
        const output = new Array(kernels.length);

        for(let i = 0; i < kernels.length; i++) {
            const out = Matrix.convolve(src, kernels[i]);
            output[i] = Matrix.addScalar(out, b[i]);
        }

        return output;
    }
    static convForwardMultiChannel(src, kernels, b) {
        const output = new Array(kernels.length);
        const outRows = src[0].length - kernels[0][0].length + 1;
        const outCols = src[0][0].length - kernels[0][0][0].length + 1;

        for(let i = 0; i < kernels.length; i++) {
            let featMap = Array.from(
                { length: outRows },
                () => new Float64Array(outCols)
            );
            for(let j = 0; j < src.length; j++) {
                const part = Matrix.convolve(src[j], kernels[i][j]);
                featMap = Matrix.add(featMap, part);
            }
            output[i] = Matrix.addScalar(featMap, b[i]);
        }

        return output;
    }
    static pool(t) {
        const output = new Array(t.length);

        for(let k = 0; k < t.length; k++) {
            const out = Array.from(
                { length: t[0].length / 2 }, 
                () => new Float64Array(t[0][0].length / 2)
            );
            for(let i = 0; i + 1 < t[0].length; i += 2) {
                for(let j = 0; j + 1 < t[0][0].length; j += 2) {
                    out[i / 2][j / 2] = Math.max(
                        t[k][i][j], t[k][i][j + 1],
                        t[k][i + 1][j], t[k][i + 1][j + 1]
                    );
                }
            }
            output[k] = out;
        }
        
        return output;
    }
    static convBackward(unpooled, pooled, activationGrad) {
        const dL_dz = new Array(unpooled.length);
        for(let k = 0; k < unpooled.length; k++) {
            const mdL_dz = Array.from(
                { length: unpooled[0].length },
                () => new Float64Array(unpooled[0][0].length)
            );
            for(let i = 0; i + 1 < unpooled[0].length; i += 2) {
                for(let j = 0; j + 1 < unpooled[0][0].length; j += 2) {
                    switch(pooled[k][i / 2][j / 2]) {
                        case unpooled[k][i][j]:
                            mdL_dz[i][j] = 1 * activationGrad[k][i][j];
                        break;
                        case unpooled[k][i][j + 1]:
                            mdL_dz[i][j + 1] = 1 * activationGrad[k][i][j + 1];
                        break;
                        case unpooled[k][i + 1][j]:
                            mdL_dz[i + 1][j] = 1 * activationGrad[k][i + 1][j];
                        break;
                        case unpooled[k][i + 1][j + 1]:
                            mdL_dz[i + 1][j + 1] = 1 * activationGrad[k][i + 1][j + 1];
                        break;
                    }
                }
            }
        }

        return dL_dz;
    }
    static backwardPass(W_L, delta_L, da_l) {
        const W_Lt = Matrix.transpose(W_L);
        const inter = Matrix.matrixVector(W_Lt, delta_L);
        return Vector.elementwise(inter, da_l);
    }
    static gradientDescentVec(v, dv, learningRate) {
        Shared.validateStructure(v, dv);
        Shared.validateSameLength(v, dv);
        
        for(let i = 0; i < v.length; i++) {
            v[i] -= dv[i] * learningRate;
        }
    }
    static gradientDescentMat(m, dm, learningRate) {
        Shared.validateStructureMat(m, dm);
        Shared.validateSameShape(m, dm);

        for(let i = 0; i < m.length; i++) {
            for(let j = 0; j < m[0].length; j++) {
                m[i][j] -= dm[i][j] * learningRate;
            }
        }
    }
}
