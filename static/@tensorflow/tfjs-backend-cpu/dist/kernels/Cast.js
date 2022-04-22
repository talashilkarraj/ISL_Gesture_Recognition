/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { Cast, util } from '@tensorflow/tfjs-core';
import { createSimpleBinaryKernelImpl } from '../utils/binary_impl';
import { zeros } from '../utils/zeros_impl';
import { complex } from './Complex';
import { identity } from './Identity';
import { real } from './Real';
export function cast(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dtype } = attrs;
    // Casting to complex64.
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return identity({ inputs: { x }, backend });
        }
        const zerosTensorInfo = zeros(backend, x.shape, x.dtype);
        const floatX = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
        const result = complex({ inputs: { real: floatX, imag: zerosTensorInfo }, backend });
        backend.disposeIntermediateTensorInfo(zerosTensorInfo);
        backend.disposeIntermediateTensorInfo(floatX);
        return result;
    }
    // Casting from complex64
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const result = cast({ inputs: { x: realPart }, backend, attrs: { dtype } });
        backend.disposeIntermediateTensorInfo(realPart);
        return result;
    }
    if (!util.hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        const result = identity({ inputs: { x }, backend });
        return { dataId: result.dataId, shape: result.shape, dtype };
    }
    if (dtype === 'int32') {
        const values = backend.data.get(x.dataId).values;
        const resultValues = Int32Array.from(values);
        return backend.makeTensorInfo(x.shape, 'int32', resultValues);
    }
    if (dtype === 'bool') {
        // This is essentially the result of notEqual(x, 0). We avoid using
        // kernel notEqual to avoid circular dependency, i.e. binary_utils ->
        // cast -> notEqual -> binary_utils.
        const xVals = backend.data.get(x.dataId).values;
        const zero = util.toTypedArray([0], x.dtype);
        const [resultData, resultShape] = createSimpleBinaryKernelImpl((a, b) => (a !== b) ? 1 : 0)(x.shape, [], xVals, zero, 'bool');
        return backend.makeTensorInfo(resultShape, 'bool', resultData);
    }
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}
export const castConfig = {
    kernelName: Cast,
    backendName: 'cpu',
    kernelFunc: cast
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ2FzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvQ2FzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsSUFBSSxFQUEyRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUcxSCxPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUNsRSxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFFMUMsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNsQyxPQUFPLEVBQUMsUUFBUSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFNUIsTUFBTSxVQUFVLElBQUksQ0FDaEIsSUFBcUU7SUFFdkUsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLEtBQUssRUFBQyxHQUFHLEtBQUssQ0FBQztJQUV0Qix3QkFBd0I7SUFDeEIsSUFBSSxLQUFLLEtBQUssV0FBVyxFQUFFO1FBQ3pCLElBQUksQ0FBQyxDQUFDLEtBQUssS0FBSyxXQUFXLEVBQUU7WUFDM0IsT0FBTyxRQUFRLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUMsRUFBRSxPQUFPLEVBQUMsQ0FBQyxDQUFDO1NBQ3pDO1FBRUQsTUFBTSxlQUFlLEdBQUcsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUN6RCxNQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsRUFBQyxNQUFNLEVBQUUs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QzFFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxVQUFVLEdBQWlCO0lBQ3RDLFVBQVUsRUFBRSxJQUFJO0lBQ2hCLFdBQVcsRUFBRSxLQUFLO0lBQ2xCLFVBQVUsRUFBRSxJQUF3QjtDQUNyQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtDYXN0LCBDYXN0QXR0cnMsIENhc3RJbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgVHlwZWRBcnJheSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHtjcmVhdGVTaW1wbGVCaW5hcnlLZXJuZWxJbXBsfSBmcm9tICcuLi91dGlscy9iaW5hcnlfaW1wbCc7XG5pbXBvcnQge3plcm9zfSBmcm9tICcuLi91dGlscy96ZXJvc19pbXBsJztcblxuaW1wb3J0IHtjb21wbGV4fSBmcm9tICcuL0NvbXBsZXgnO1xuaW1wb3J0IHtpZGVudGl0eX0gZnJvbSAnLi9JZGVudGl0eSc7XG5pbXBvcnQge3JlYWx9IGZyb20gJy4vUmVhbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBjYXN0KFxuICAgIGFyZ3M6IHtpbnB1dHM6IENhc3RJbnB1dHMsIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLCBhdHRyczogQ2FzdEF0dHJzfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHt4fSA9IGlucHV0cztcbiAgY29uc3Qge2R0eXBlfSA9IGF0dHJzO1xuXG4gIC8vIENhc3RpbmcgdG8gY29tcGxleDY0LlxuICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgaWYgKHguZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICByZXR1cm4gaWRlbnRpdHkoe2lucHV0czoge3h9LCBiYWNrZW5kfSk7XG4gICAgfVxuXG4gICAgY29uc3QgemVyb3NUZW5zb3JJbmZvID0gemVyb3MoYmFja2VuZCwgeC5zaGFwZSwgeC5kdHlwZSk7XG4gICAgY29uc3QgZmxvYXRYID0gY2FzdCh7aW5wdXRzOiB7eH0sIGJhY2tlbmQsIGF0dHJzOiB7ZHR5cGU6ICdmbG9hdDMyJ319KTtcblxuICAgIGNvbnN0IHJlc3VsdCA9XG4gICAgICAgIGNvbXBsZXgoe2lucHV0czoge3JlYWw6IGZsb2F0WCwgaW1hZzogemVyb3NUZW5zb3JJbmZvfSwgYmFja2VuZH0pO1xuXG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh6ZXJvc1RlbnNvckluZm8pO1xuICAgIGJhY2tlbmQuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oZmxvYXRYKTtcblxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICAvLyBDYXN0aW5nIGZyb20gY29tcGxleDY0XG4gIGlmICh4LmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgIGNvbnN0IHJlYWxQYXJ0ID0gcmVhbCh7aW5wdXRzOiB7aW5wdXQ6IHh9LCBiYWNrZW5kfSk7XG4gICAgY29uc3QgcmVzdWx0ID0gY2FzdCh7aW5wdXRzOiB7eDogcmVhbFBhcnR9LCBiYWNrZW5kLCBhdHRyczoge2R0eXBlfX0pO1xuXG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZWFsUGFydCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgaWYgKCF1dGlsLmhhc0VuY29kaW5nTG9zcyh4LmR0eXBlLCBkdHlwZSkpIHtcbiAgICAvLyBXZSBkb24ndCBjaGFuZ2UgdGhlIHVuZGVybHlpbmcgZGF0YSwgc2luY2Ugd2UgY2FzdCB0byBoaWdoZXJcbiAgICAvLyBwcmVjaXNpb24uXG4gICAgY29uc3QgcmVzdWx0ID0gaWRlbnRpdHkoe2lucHV0czoge3h9LCBiYWNrZW5kfSk7XG4gICAgcmV0dXJuIHtkYXRhSWQ6IHJlc3VsdC5kYXRhSWQsIHNoYXBlOiByZXN1bHQuc2hhcGUsIGR0eXBlfTtcbiAgfVxuXG4gIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIGNvbnN0IHZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IEludDMyQXJyYXkuZnJvbSh2YWx1ZXMpO1xuICAgIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKHguc2hhcGUsICdpbnQzMicsIHJlc3VsdFZhbHVlcyk7XG4gIH1cblxuICBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgIC8vIFRoaXMgaXMgZXNzZW50aWFsbHkgdGhlIHJlc3VsdCBvZiBub3RFcXVhbCh4LCAwKS4gV2UgYXZvaWQgdXNpbmdcbiAgICAvLyBrZXJuZWwgbm90RXF1YWwgdG8gYXZvaWQgY2lyY3VsYXIgZGVwZW5kZW5jeSwgaS5lLiBiaW5hcnlfdXRpbHMgLT5cbiAgICAvLyBjYXN0IC0+IG5vdEVxdWFsIC0+IGJpbmFyeV91dGlscy5cbiAgICBjb25zdCB4VmFscyA9IGJhY2tlbmQuZGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICAgIGNvbnN0IHplcm8gPSB1dGlsLnRvVHlwZWRBcnJheShbMF0sIHguZHR5cGUpO1xuXG4gICAgY29uc3QgW3Jlc3VsdERhdGEsIHJlc3VsdFNoYXBlXSA9IGNyZWF0ZVNpbXBsZUJpbmFyeUtlcm5lbEltcGwoXG4gICAgICAgIChhLCBiKSA9PiAoYSAhPT0gYikgPyAxIDogMCkoeC5zaGFwZSwgW10sIHhWYWxzLCB6ZXJvLCAnYm9vbCcpO1xuXG4gICAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8ocmVzdWx0U2hhcGUsICdib29sJywgcmVzdWx0RGF0YSk7XG4gIH1cblxuICB0aHJvdyBuZXcgRXJyb3IoYEVycm9yIGluIENhc3Q6IGZhaWxlZCB0byBjYXN0ICR7eC5kdHlwZX0gdG8gJHtkdHlwZX1gKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNhc3RDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQ2FzdCxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiBjYXN0IGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=