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
    static validateStructure3DTensor(a, b) {
        if(
            !Array.isArray(a) ||
            !Array.isArray(a[0]) ||
            !isArrayBufferView(a[0][0]) ||
            !Array.isArray(b) ||
            !Array.isArray(b[0]) ||
            !isArrayBufferView(b[0][0])
        ) throw new TypeError("Both tensors must be a 3D Float64Array");
        if(
            !a.length ||
            !a[0].length ||
            !a[0][0].length ||
            !b.length ||
            !b[0].length ||
            !b[0][0].length
        ) throw new TypeError("Both 3D tensors must have positive length in all dimensions");

        return true;
    }
    static validateStructure3DTensorSingle(a) {
        if(
            !Array.isArray(a) ||
            !Array.isArray(a[0]) ||
            !isArrayBufferView(a[0][0])
        ) throw new TypeError("Tensor must be a 3D Float64Array");
        if(
            !a.length ||
            !a[0].length ||
            !a[0][0].length
        ) throw new TypeError("Tensor must have positive length in all dimensions");

        return true;
    }
    static validateSameDimensions3D(a, b) {
        if(
            a.length !== b.length ||
            a[0].length !== b[0].length ||
            a[0][0].length !== b[0][0].length
        ) throw new TypeError("Both 3D tensors must have the same dimensions");

        return true;
    }
    static validateStructureTensor(a, b) {
        if(
            !Array.isArray(a) ||
            !Array.isArray(a[0]) ||
            !Array.isArray(a[0][0]) ||
            !isArrayBufferView(a[0][0][0]) ||
            !Array.isArray(b) ||
            !Array.isArray(b[0]) ||
            !Array.isArray(b[0][0]) ||
            !isArrayBufferView(b[0][0][0])
        ) throw new TypeError("Both tensors must be a 4D Float64Array");
        if(
            !a.length ||
            !a[0].length ||
            !a[0][0].length ||
            !a[0][0][0].length ||
            !b.length ||
            !b[0].length ||
            !b[0][0].length ||
            !b[0][0][0].length
        ) throw new TypeError("Both 4D tensors must have positive length in all dimensions");

        return true;
    }
    static validateStructureTensorSingle(a) {
        if(
            !Array.isArray(a) ||
            !Array.isArray(a[0]) ||
            !Array.isArray(a[0][0]) ||
            !isArrayBufferView(a[0][0][0])
        ) throw new TypeError("Tensor must be a 4D Float64Array");
        if(
            !a.length ||
            !a[0].length ||
            !a[0][0].length ||
            !a[0][0][0].length
        ) throw new TypeError("Tensor must have positive length in all dimensions");

        return true;
    }
    static validateSameDimensions(a, b) {
        if(
            a.length !== b.length ||
            a[0].length !== b[0].length ||
            a[0][0].length !== b[0][0].length ||
            a[0][0][0].length !== b[0][0][0].length
        ) throw new TypeError("Both tensors must have the same dimensions");

        return true;
    }
}

export class Vector {
    static create(length, min, max) {
        if(length <= 0 || typeof length !== "number")
            throw new Error("Length parameter must be a positive number");

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
            throw new TypeError("The columns of A must be equal to the rows of B for matrix multiplication. Try transposing a matrix or flipping the order of arguments");

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
            throw new TypeError("Vector length must be the same as matrix column length for matrix vector multiplication");

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
    static divScalar(m, s) {
        Shared.validateStructureMatSingle(m);

        const mat = new Array(m.length);
        for(let i = 0; i < m.length; i++) {
            mat[i] = new Float64Array(m[0].length);
            for(let j = 0; j < m[0].length; j++) {
                mat[i][j] = m[i][j] / s;
            }
        }

        return mat;
    }
    static addBroadcast(m, v) {
        Shared.validateStructureMatSingle(m);
        Shared.validateStructureSingle(v);
        if(v.length !== m[0].length)
            throw new TypeError("Vector length must be the same as matrix column length for matrix vector multiplication");

        const mat = new Array(m.length);
        for(let i = 0; i < m.length; i++) {
            mat[i] = new Float64Array(m[0].length);
            for(let j = 0; j < m[0].length; j++) {
                mat[i][j] = m[i][j] + v[j]; 
            }
        }

        return mat;
    }

    static crossCorrelate(src, kernel) {
        Shared.validateStructureMatSingle(src);
        Shared.validateStructureMatSingle(kernel);
        if(kernel.length > src.length || kernel[0].length > src[0].length)
            throw new TypeError("Kernel cannot have larger size than source matrix");

        const outRows = src.length - kernel.length + 1;
        const outCols = src[0].length - kernel[0].length + 1;
        const mat = new Array(outRows);

        for(let i = 0; i < outRows; i++) {
            mat[i] = new Float64Array(outCols);
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
    static fullCrossCorrelate(src, kernel) {
        Shared.validateStructureMatSingle(src);
        Shared.validateStructureMatSingle(kernel);

        const padVertical = kernel.length - 1;
        const padHorizontal = kernel[0].length - 1;
        const paddedRows = src.length + (2 * padVertical);
        const paddedCols = src[0].length + (2 * padHorizontal);
        const padded = Array.from({ length: paddedRows }, () => new Float64Array(paddedCols));
        for(let i = 0; i < src.length; i++) {
            for(let j = 0; j < src[0].length; j++) {
                padded[i + padVertical][j + padHorizontal] = src[i][j];
            }
        }
        
        const outRows = src.length + kernel.length - 1;
        const outCols = src[0].length + kernel[0].length - 1;
        const mat = new Array(outRows);
        for(let i = 0; i < outRows; i++) {
            mat[i] = new Float64Array(outCols);
            for(let j = 0; j < outCols; j++) {
                let sum = 0;
                for(let k = 0; k < kernel.length; k++) {
                    for(let l = 0; l < kernel[0].length; l++) {
                        sum += padded[i + k][j + l] * kernel[k][l];
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

        const flipped = new Array(m.length);
        const rows = m.length - 1;
        const cols = m[0].length - 1;
        for(let i = 0; i < m.length; i++) {
            flipped[i] = new Float64Array(m[0].length);
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

export class Tensor4D {
    static create(ch_out, ch_in, height, width, min, max) {
        if(
            ch_out <= 0 || typeof ch_out !== "number" ||
            ch_in <= 0 || typeof ch_in !== "number"
        )
            throw new Error("Channel parameters must be a positive number");

        const tensor = new Array(ch_out);
        for(let i = 0; i < ch_out; i++) {
            tensor[i] = new Array(ch_in);
            for(let j = 0; j < ch_in; j++) {
                tensor[i][j] = Matrix.create(height, width, min, max);
            }
        }

        return tensor;
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
        
        return rand * (max - min) + min;
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

        return -((expected * Math.log(p)) + ((1 - expected) * (Math.log(p2))));
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
        Shared.validateStructure3DTensorSingle(z);

        const a = new Array(z.length);
        for(let i = 0; i < z.length; i++) {
            a[i] = new Array(z[0].length);
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
        Shared.validateStructure3DTensor(a, z);
        Shared.validateSameDimensions3D(a, z);

        const da_dz = new Array(z.length);
        for(let i = 0; i < z.length; i++) {
            da_dz[i] = new Array(z[0].length);
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
    static convForward(src, filters, b) {
        Shared.validateStructure3DTensorSingle(src);
        Shared.validateStructureTensorSingle(filters);
        Shared.validateStructureSingle(b);

        const output = new Array(filters.length);
        const outRows = src[0].length - filters[0][0].length + 1;
        const outCols = src[0][0].length - filters[0][0][0].length + 1;

        for(let i = 0; i < filters.length; i++) {
            let featMap = Array.from(
                { length: outRows },
                () => new Float64Array(outCols)
            );
            for(let j = 0; j < src.length; j++) {
                const part = Matrix.crossCorrelate(src[j], filters[i][j]);
                featMap = Matrix.add(featMap, part);
            }
            output[i] = Matrix.addScalar(featMap, b[i]);
        }

        return output;
    }
    static pool(t) {
        Shared.validateStructure3DTensorSingle(t);

        const output = new Array(t.length);

        for(let k = 0; k < t.length; k++) {
            const out = new Array(t[0].length / 2);
            for(let i = 0; i + 1 < t[0].length; i += 2) {
                out[i / 2] = new Float64Array(t[0][0].length / 2);
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
    static convBackward(unpooled, pooled, dL_da_pool, activationGrad) {
        Shared.validateStructure3DTensor(pooled, dL_da_pool);
        Shared.validateSameDimensions3D(pooled, dL_da_pool);
        Shared.validateStructure3DTensor(unpooled, activationGrad);
        Shared.validateSameDimensions3D(unpooled, activationGrad);

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
                            mdL_dz[i][j] = dL_da_pool[k][i / 2][j / 2] * activationGrad[k][i][j];
                        break;
                        case unpooled[k][i][j + 1]:
                            mdL_dz[i][j + 1] = dL_da_pool[k][i / 2][j / 2] * activationGrad[k][i][j + 1];
                        break;
                        case unpooled[k][i + 1][j]:
                            mdL_dz[i + 1][j] = dL_da_pool[k][i / 2][j / 2] * activationGrad[k][i + 1][j];
                        break;
                        case unpooled[k][i + 1][j + 1]:
                            mdL_dz[i + 1][j + 1] = dL_da_pool[k][i / 2][j / 2] * activationGrad[k][i + 1][j + 1];
                        break;
                    }
                }
            }
            dL_dz[k] = mdL_dz;
        }

        return dL_dz;
    }
    static convBackwardPrev(W_L, delta_L) {
        Shared.validateStructureTensorSingle(W_L);
        Shared.validateStructure3DTensorSingle(delta_L);

        const outRows = W_L[0][0].length + delta_L[0].length - 1;
        const outCols = W_L[0][0][0].length + delta_L[0][0].length - 1;
        const delta_l = new Array(W_L[0].length);

        for(let i = 0; i < W_L[0].length; i++) {
            let featMap = Array.from(
                { length: outRows },
                () => new Float64Array(outCols)
            );
            for(let j = 0; j < delta_L.length; j++) {
                const part = Matrix.fullCrossCorrelate(delta_L[j], Matrix.flip(W_L[j][i]));
                featMap = Matrix.add(featMap, part);
            }
            delta_l[i] = featMap;
        }

        return delta_l;
    }
    static convKernelGrad(input_l, dL_dz) {
        Shared.validateStructure3DTensorSingle(dL_dz);
        Shared.validateStructure3DTensorSingle(input_l);

        const dz_dk = new Array(dL_dz.length);  
        for(let i = 0; i < dL_dz.length; i++) {
            dz_dk[i] = new Array(input_l.length);
            for(let j = 0; j < input_l.length; j++) {
                const kernelGrad = Matrix.crossCorrelate(input_l[j], dL_dz[i]);
                dz_dk[i][j] = kernelGrad;
            }
        }

        return dz_dk;
    }
    static convBiasGrad(dL_dz) {
        Shared.validateStructure3DTensorSingle(dL_dz);

        const vec = new Float64Array(dL_dz.length);
        for(let i = 0; i < dL_dz.length; i++) {
            let sum = 0;
            for(let j = 0; j < dL_dz[0].length; j++) {
                for(let k = 0; k < dL_dz[0][0].length; k++) {
                    sum += dL_dz[i][j][k];
                }
            }
            vec[i] = sum;
        }

        return vec;
    }
    static globalAveragePool(t) {
        Shared.validateStructure3DTensorSingle(t);

        const vec = new Float64Array(t.length);
        for(let i = 0; i < t.length; i++) {
            let sum = 0;
            for(let j = 0; j < t[0].length; j++) {
                for(let k = 0; k < t[0][0].length; k++) {
                    sum += t[i][j][k];
                }
            }
            sum /= t[0].length * t[0][0].length;
            vec[i] = sum;
        }

        return vec;
    }
    static convBackwardGAP(
        delta_L, W_L, 
        unpooled_channels, unpooled_rows, 
        unpooled_cols, activationGrad_l
    ) {
        Shared.validateStructure3DTensorSingle(activationGrad_l);
        
        const delta_l = Matrix.matrixVector(Matrix.transpose(W_L), delta_L);
        const dL_dz = new Array(unpooled_channels);
        for(let i = 0; i < delta_l.length; i++) {
            dL_dz[i] = new Array(unpooled_rows);
            const grad = delta_l[i] / (unpooled_cols * unpooled_rows);
            for(let j = 0; j < unpooled_rows; j++) {
                dL_dz[i][j] = new Float64Array(unpooled_cols)
                for(let k = 0; k < unpooled_cols; k++) {
                    dL_dz[i][j][k] = grad * activationGrad_l[i][j][k];
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
    static convGradientDescent(w, dw, learningRate) {
        Shared.validateStructureTensor(w, dw);

        const A = w.length;
        const B = w[0].length;
        const C = w[0][0].length;
        const D = w[0][0][0].length;
        const total = A * B * C * D;

        for(let i = 0; i < total; i++) {
            const f = Math.floor(i / (B * C * D));
            const c = Math.floor((i / (C * D))) % B;
            const h = Math.floor((i / D)) % C;
            const x = i % D;

            w[f][c][h][x] -= dw[f][c][h][x] * learningRate;
        }
    }
    // crude tokenization; no fallback or fancy split algorithm. Ensure that every possible token exists in the vocabulary.
    static tokenize(spaceSeparatedInput, vocab, embed, pos) {
        const prepended = "<CLS>" + " " + spaceSeparatedInput;
        const sentence = prepended.split(" ");
        const tokenized = new Array(sentence.length);
        for(let i = 0; i < sentence.length; i++) {
            tokenized[i] = Vector.add(
                embed[vocab[`${sentence[i]}`]],
                pos[i]
            );
        }

        return tokenized;
    }
    static initSinusoidal(max_seq, d_model) {
        const mat = Array.from(
            { length: max_seq },
            () => new Float64Array(d_model)
        );

        for(let i = 0; i < mat.length; i++) {
            for(let j = 0; j < mat[0].length; j++) {
                mat[i][j] =
                    Math.sin( (i / Math.pow(10000, (2 * (~~(j / 2))) / d_model)) ) * ((j + 1) % 2)
                    + Math.cos( (i / Math.pow(10000, (2 * (~~(j / 2))) / d_model)) ) * ((j + 2) % 2);
            }
        }

        return mat;
    }
    static attention(embeddings, Wq, Wk, Wv, d_k) {
        const cache = {
            Q: null,
            K: null,
            V: null,
            QK_T: null,
            A: null,
        };
        const Q = Matrix.multiply(embeddings, Wq);
        const K = Matrix.multiply(embeddings, Wk);
        const V = Matrix.multiply(embeddings, Wv);
        cache.Q = Q;
        cache.K = K;
        cache.V = V;

        const QK_T = Matrix.divScalar(
            Matrix.multiply(Q, Matrix.transpose(K)),
            d_k,
        );
        cache.QK_T = QK_T;

        const A = new Array(QK_T.length);
        for(let i = 0; i < QK_T.length; i++) {
            A[i] = Activations.softmax(QK_T[i]);
        }
        cache.A = A;

        return [
            Matrix.multiply(A, V), 
            cache,
        ];
    }
    static layerNorm(residual, scale, shift) {
        const normalized = new Array(residual.length);
        const scaledShifted = new Array(residual.length);
        const means_var = Array.from(
            { length: residual.length },
            () => new Float64Array(2)
        );
        for(let i = 0; i < residual.length; i++) {
            let sum = 0;
            for(let j = 0; j < residual[0].length; j++) {
                sum += residual[i][j];
            }
            means_var[i][0] = sum / residual[0].length;
        }
        for(let i = 0; i < residual.length; i++) {
            let sum = 0;
            for(let j = 0; j < residual[0].length; j++) {
                sum += Math.pow(residual[i][j] - means_var[i][0], 2);
            }
            means_var[i][1] = sum / residual[0].length;
        }

        for(let i = 0; i < residual.length; i++) {
            normalized[i] = new Float64Array(residual[0].length);
            for(let j = 0; j < residual[0].length; j++) {
                normalized[i][j] = (residual[i][j] - means_var[i][0]) / Math.sqrt(means_var[i][1] + 1e-8);
            }
        }
        for(let i = 0; i < residual.length; i++) {
            scaledShifted[i] = new Float64Array(residual[0].length);
            for(let j = 0; j < residual[0].length; j++) {
                scaledShifted[i][j] = scale[j] * normalized[i][j] + shift[j];
            }
        }

        return [
            scaledShifted,
            {
                CLS_normalized: normalized[0],
                mean: means_var[0][0],
                var: means_var[0][1],
            }
        ];
    }
    static transformerFFN(input, W, b) {
        return Matrix.addBroadcast(
            Matrix.multiply(input, W),
            b
        );
    }

    static LNbackward(dL_dh, clsCache, varianceCache, scale, d_k) {
        const scale_grad = Vector.elementwise(dL_dh, clsCache);
        const shift_grad = dL_dh;
        const dL_dr = new Float64Array(dL_dh.length);
        let dL_dh_mean = 0;
        let dL_dh_var = 0;
        for(let i = 0; i < dL_dh.length; i++) {
            dL_dh_mean += dL_dh[i];
            dL_dh_var += dL_dh[i] * clsCache[i]
        }
        dL_dh_mean /= d_k;
        dL_dh_var /= d_k;
        dL_dh_var = Vector.elementwise(
            clsCache,
            Float64Array.from({ length: dL_dh.length }, () => dL_dh_var)
        );
        for(let i = 0; i < dL_dh.length; i++) {
            dL_dr[i] = 
                (scale[i] / Math.sqrt(varianceCache + 1e-8))
                * (dL_dh[i] - dL_dh_mean - dL_dh_var[i]);
        }

        return [scale_grad, shift_grad, dL_dr];
    }
    static FFNbackward(
        dL_dr,
        FFN_a1_CLS,
        activationGrad,
        attention_postLN_CLS,
        W1,
        W2,
    ) {
        // Because row vectors are implicitly treated as column vectors, all the transposes below need to be reversed in order for the dimensions to match.
        // The math itself is consistent, even with reversed transposes; this is a work-around only for the code.
        const b2_grad = dL_dr;
        const W2_grad = Vector.outer(FFN_a1_CLS, dL_dr);
        const dL_dz1 = NeuralNet.backwardPass(
            Matrix.transpose(W2), dL_dr, 
            activationGrad
        );
        const b1_grad = dL_dz1;
        const W1_grad = Vector.outer(attention_postLN_CLS, dL_dz1);
        const dL_dxFFN = Matrix.matrixVector(W1, dL_dz1);

        return [
            b2_grad,
            W2_grad,
            b1_grad,
            W1_grad,
            dL_dxFFN
        ];
    }
    static classificationHeadBackward(dL_dz, CLS, Wc) {
        const bc_grad = dL_dz;
        const Wc_grad = Vector.outer(dL_dz, CLS);
        const dL_dh2 = Matrix.matrixVector(Matrix.transpose(Wc), dL_dz);

        return [
            bc_grad,
            Wc_grad,
            dL_dh2
        ];
    }
    static attentionBackward(
        embeddings,
        dL_dr_v,
        cacheA,
        cacheQ,
        cacheK,
        cacheV,
        Wq,
        Wk,
        Wv,
        d_k,
    ) {
        const dL_dr = new Array(embeddings.length);
        for(let i = 0; i < dL_dr.length; i++) {
            dL_dr[i] = i === 0 ? dL_dr_v : new Float64Array(d_k);
        }

        const dL_dV = Matrix.multiply(
            Matrix.transpose(cacheA),
            dL_dr,
        );
        const dL_da = Matrix.multiply(
            dL_dr,
            Matrix.transpose(cacheV),
        );
        const dL_ds = new Array(dL_da.length);
        for(let i = 0; i < dL_da.length; i++) {
            const dot = Vector.dot(dL_da[i], cacheA[i])
            dL_ds[i] = Vector.elementwise(
                cacheA[i],
                Vector.subtract(
                    dL_da[i],
                    Float64Array.from(
                        { length: dL_da[0].length },
                        () => dot
                    )
                )
            );
        }
        const dL_dQKt = Matrix.divScalar(dL_ds, d_k);
        const dL_dQ = Matrix.multiply(dL_dQKt, cacheK);
        const dL_dK = Matrix.multiply(Matrix.transpose(dL_dQKt), cacheQ);

        const Wq_grad = Matrix.multiply(Matrix.transpose(embeddings), dL_dQ);
        const Wk_grad = Matrix.multiply(Matrix.transpose(embeddings), dL_dK);
        const Wv_grad = Matrix.multiply(Matrix.transpose(embeddings), dL_dV);
        const dL_dxAttention = Matrix.add(
            Matrix.multiply(
                dL_dQ, Matrix.transpose(Wq)
            ),
            Matrix.add(
                Matrix.multiply(
                    dL_dK, Matrix.transpose(Wk)
                ),
                Matrix.multiply(
                    dL_dV, Matrix.transpose(Wv)
                ),
            )
        );

        return [
            Wq_grad,
            Wk_grad,
            Wv_grad,
            dL_dxAttention,
            dL_dr
        ];
    }
    static embeddingBackward(dL_dx, spaceSeparatedInput, E, d_k, vocab) {
        const E_grad = Array.from({ length: E.length }, () => new Float64Array(d_k));
        const sentence = ("<CLS>" + " " + spaceSeparatedInput).split(" ");
        for(let i = 0; i < sentence.length; i++) {
            E_grad[vocab[`${sentence[i]}`]] = Vector.add(
                E_grad[vocab[`${sentence[i]}`]],
                dL_dx[i]
            );
        }

        return E_grad;
    }
}
