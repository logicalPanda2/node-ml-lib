export class Statistics {
    constructor() {
        this.total = 0;
        this.successes = 0;
        this.fails = 0;
    }
}

export function test(description, stats, boolCallback) {
    if(boolCallback() === true) {
        console.log(`    ${description}: \x1b[32mPASS\x1b[0m`);
        stats.successes++;
    } else {
        console.log(`    ${description}: \x1b[31mFAIL\x1b[0m`);
        stats.fails++;
    }
    stats.total++;
}
export function summary(stats) {
    console.log(`${stats.total} TOTAL TESTS`);
    console.log(`\x1b[32m${stats.successes} PASSED\x1b[0m`);
    console.log(`\x1b[31m${stats.fails} FAILED\x1b[0m`);
}
