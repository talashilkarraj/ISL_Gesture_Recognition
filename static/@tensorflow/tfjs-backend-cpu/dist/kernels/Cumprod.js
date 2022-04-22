/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { backend_util, Cumprod, upcastType, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { transpose } from './Transpose';
export function cumprod(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, exclusive, reverse } = attrs;
    assertNotComplex(x, 'cumprod');
    const permutation = backend_util.getAxesPermutation([axis], x.shape.length);
    let $x = x;
    if (permutation != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
    }
    const permutedAxis = backend_util.getInnerMostAxes(1, x.shape.length)[0];
    if (permutedAxis !== $x.shape.length - 1) {
        throw new Error(`backend.cumprod in CPU expects an inner-most ` +
            `axis=${$x.shape.length - 1} but got axis=${permutedAxis}`);
    }
    const resultDtype = upcastType($x.dtype, 'int32');
    const vals = util.makeOnesTypedArray(util.sizeFromShape($x.shape), resultDtype);
    const aVals = backend.data.get($x.dataId).values;
    const finalDim = $x.shape[$x.shape.length - 1];
    const indexAdjuster = reverse ?
        (i, j) => i + finalDim - j - 1 :
        (i, j) => i + j;
    for (let i = 0; i < aVals.length; i += finalDim) {
        for (let j = 0; j < finalDim; j++) {
            const idx = indexAdjuster(i, j);
            if (j === 0) {
                vals[idx] = exclusive ? 1 : aVals[idx];
            }
            else {
                const prevIdx = indexAdjuster(i, j - 1);
                vals[idx] = exclusive ? aVals[prevIdx] * vals[prevIdx] :
                    aVals[idx] * vals[prevIdx];
            }
        }
    }
    const result = backend.makeTensorInfo($x.shape, resultDtype, vals);
    if (permutation != null) {
        const reversePermutation = backend_util.getUndoAxesPermutation(permutation);
        const reverseTransposedResult = transpose({ inputs: { x: result }, backend, attrs: { perm: reversePermutation } });
        backend.disposeIntermediateTensorInfo(result);
        backend.disposeIntermediateTensorInfo($x);
        return reverseTransposedResult;
    }
    return result;
}
export const cumprodConfig = {
    kernelName: Cumprod,
    backendName: 'cpu',
    kernelFunc: cumprod
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ3VtcHJvZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvQ3VtcHJvZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFFLE9BQU8sRUFBaUYsVUFBVSxFQUFFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRzdKLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUM3QyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXRDLE1BQU0sVUFBVSxPQUFPLENBQ25CLElBQzJCO0lBQzdCLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25CLE1BQU0sRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBQyxHQUFHLEtBQUssQ0FBQztJQUV6QyxnQkFBZ0IsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFL0IsTUFBTSxXQUFXLEdBQUcsWUFBWSxDQUFDLGtCQUFrQixDQUFDLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUM1RSxJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDWCxJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7UUFDdkIsRUFBRSxHQUFHLFNBQVMsQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUUsV0FBVyxFQUFDLEVBQUMsQ0FBQyxDQUFDO0tBQ3BFO0lBQ0QsTUFBTSxZQUFZLEdBQUcsWUFBWSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFB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QUFDaEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBaUI7SUFDekMsVUFBVSxFQUFFLE9BQU87SUFDbkIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLE9BQTJCO0NBQ3hDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBDdW1wcm9kLCBDdW1wcm9kQXR0cnMsIEN1bXByb2RJbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgVHlwZWRBcnJheSwgdXBjYXN0VHlwZSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHthc3NlcnROb3RDb21wbGV4fSBmcm9tICcuLi9jcHVfdXRpbCc7XG5pbXBvcnQge3RyYW5zcG9zZX0gZnJvbSAnLi9UcmFuc3Bvc2UnO1xuXG5leHBvcnQgZnVuY3Rpb24gY3VtcHJvZChcbiAgICBhcmdzOiB7aW5wdXRzOiBDdW1wcm9kSW5wdXRzLCBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSxcbiAgICAgICAgICAgYXR0cnM6IEN1bXByb2RBdHRyc30pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7YXhpcywgZXhjbHVzaXZlLCByZXZlcnNlfSA9IGF0dHJzO1xuXG4gIGFzc2VydE5vdENvbXBsZXgoeCwgJ2N1bXByb2QnKTtcblxuICBjb25zdCBwZXJtdXRhdGlvbiA9IGJhY2tlbmRfdXRpbC5nZXRBeGVzUGVybXV0YXRpb24oW2F4aXNdLCB4LnNoYXBlLmxlbmd0aCk7XG4gIGxldCAkeCA9IHg7XG4gIGlmIChwZXJtdXRhdGlvbiAhPSBudWxsKSB7XG4gICAgJHggPSB0cmFuc3Bvc2Uoe2lucHV0czoge3h9LCBiYWNrZW5kLCBhdHRyczoge3Blcm06IHBlcm11dGF0aW9ufX0pO1xuICB9XG4gIGNvbnN0IHBlcm11dGVkQXhpcyA9IGJhY2tlbmRfdXRpbC5nZXRJbm5lck1vc3RBeGVzKDEsIHguc2hhcGUubGVuZ3RoKVswXTtcblxuICBpZiAocGVybXV0ZWRBeGlzICE9PSAkeC5zaGFwZS5sZW5ndGggLSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgYmFja2VuZC5jdW1wcm9kIGluIENQVSBleHBlY3RzIGFuIGlubmVyLW1vc3QgYCArXG4gICAgICAgIGBheGlzPSR7JHguc2hhcGUubGVuZ3RoIC0gMX0gYnV0IGdvdCBheGlzPSR7cGVybXV0ZWRBeGlzfWApO1xuICB9XG5cbiAgY29uc3QgcmVzdWx0RHR5cGUgPSB1cGNhc3RUeXBlKCR4LmR0eXBlLCAnaW50MzInKTtcbiAgY29uc3QgdmFscyA9IHV0aWwubWFrZU9uZXNUeXBlZEFycmF5KFxuICAgICAgICAgICAgICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZSgkeC5zaGFwZSksIHJlc3VsdER0eXBlKSBhcyBUeXBlZEFycmF5O1xuXG4gIGNvbnN0IGFWYWxzID0gYmFja2VuZC5kYXRhLmdldCgkeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCBmaW5hbERpbSA9ICR4LnNoYXBlWyR4LnNoYXBlLmxlbmd0aCAtIDFdO1xuICBjb25zdCBpbmRleEFkanVzdGVyID0gcmV2ZXJzZSA/XG4gICAgICAoaTogbnVtYmVyLCBqOiBudW1iZXIpID0+IGkgKyBmaW5hbERpbSAtIGogLSAxIDpcbiAgICAgIChpOiBudW1iZXIsIGo6IG51bWJlcikgPT4gaSArIGo7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYVZhbHMubGVuZ3RoOyBpICs9IGZpbmFsRGltKSB7XG4gICAgZm9yIChsZXQgaiA9IDA7IGogPCBmaW5hbERpbTsgaisrKSB7XG4gICAgICBjb25zdCBpZHggPSBpbmRleEFkanVzdGVyKGksIGopO1xuICAgICAgaWYgKGogPT09IDApIHtcbiAgICAgICAgdmFsc1tpZHhdID0gZXhjbHVzaXZlID8gMSA6IGFWYWxzW2lkeF07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBwcmV2SWR4ID0gaW5kZXhBZGp1c3RlcihpLCBqIC0gMSk7XG4gICAgICAgIHZhbHNbaWR4XSA9IGV4Y2x1c2l2ZSA/IGFWYWxzW3ByZXZJZHhdICogdmFsc1twcmV2SWR4XSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFWYWxzW2lkeF0gKiB2YWxzW3ByZXZJZHhdO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGNvbnN0IHJlc3VsdCA9IGJhY2tlbmQubWFrZVRlbnNvckluZm8oJHguc2hhcGUsIHJlc3VsdER0eXBlLCB2YWxzKTtcblxuICBpZiAocGVybXV0YXRpb24gIT0gbnVsbCkge1xuICAgIGNvbnN0IHJldmVyc2VQZXJtdXRhdGlvbiA9IGJhY2tlbmRfdXRpbC5nZXRVbmRvQXhlc1Blcm11dGF0aW9uKHBlcm11dGF0aW9uKTtcbiAgICBjb25zdCByZXZlcnNlVHJhbnNwb3NlZFJlc3VsdCA9IHRyYW5zcG9zZShcbiAgICAgICAge2lucHV0czoge3g6IHJlc3VsdH0sIGJhY2tlbmQsIGF0dHJzOiB7cGVybTogcmV2ZXJzZVBlcm11dGF0aW9ufX0pO1xuXG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXN1bHQpO1xuICAgIGJhY2tlbmQuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oJHgpO1xuXG4gICAgcmV0dXJuIHJldmVyc2VUcmFuc3Bvc2VkUmVzdWx0O1xuICB9XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGNvbnN0IGN1bXByb2RDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQ3VtcHJvZCxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiBjdW1wcm9kIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=