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
import { Unpack } from '@tensorflow/tfjs-core';
import { reshape } from './Reshape';
import { slice } from './Slice';
export function unpack(args) {
    const { inputs, backend, attrs } = args;
    const { value } = inputs;
    let { axis } = attrs;
    if (axis < 0) {
        axis += value.shape.length;
    }
    const valueRank = value.shape.length;
    const num = value.shape[axis];
    const outShape = new Array(valueRank - 1);
    let outIndex = 0;
    for (let i = 0; i < valueRank; i++) {
        if (i !== axis) {
            outShape[outIndex++] = value.shape[i];
        }
    }
    const begin = new Array(valueRank).fill(0);
    const size = value.shape.slice();
    size[axis] = 1;
    const res = new Array(num);
    for (let i = 0; i < res.length; i++) {
        begin[axis] = i;
        const tempRes = slice({ inputs: { x: value }, backend, attrs: { begin, size } });
        res[i] = reshape({ inputs: { x: tempRes }, backend, attrs: { shape: outShape } });
        backend.disposeIntermediateTensorInfo(tempRes);
    }
    return res;
}
export const unpackConfig = {
    kernelName: Unpack,
    backendName: 'cpu',
    kernelFunc: unpack
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVW5wYWNrLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9VbnBhY2sudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUF1QyxNQUFNLEVBQTRCLE1BQU0sdUJBQXVCLENBQUM7QUFHOUcsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNsQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRTlCLE1BQU0sVUFBVSxNQUFNLENBQ2xCLElBQXlFO0lBRTNFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsS0FBSyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ3ZCLElBQUksRUFBQyxJQUFJLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFbkIsSUFBSSxJQUFJLEdBQUcsQ0FBQyxFQUFFO1FBQ1osSUFBSSxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDO0tBQzVCO0lBRUQsTUFBTSxTQUFTLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUM7SUFFckMsTUFBTSxHQUFHLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUM5QixNQUFNLFFBQVEsR0FBYSxJQUFJLEtBQUssQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDcEQsSUFBSSxRQUFRLEdBQUcsQ0FBQyxDQUFDO0lBQ2pCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxTQUFTLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDbEMsSUFBSSxDQUFDLEtBQUssSUFBSSxFQUFFO1lBQ2QsUUFBUSxDQUFDLFFBQVEsRUFBRSxDQUFDLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUN2QztLQUNGO0lBRUQsTUFBTSxLQUFLLEdBQUcsSUFBSSxLQUFLLENBQUMsU0FBUyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLE1BQU0sSUFBSSxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLENBQUM7SUFDakMsSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUNmLE1BQU0sR0FBRyxHQUFHLElBQUksS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBQzNCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxHQUFHLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ25DLEtBQUssQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxPQUFPLEdBQUcsS0FBSyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLEtBQUssRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxLQUFLLEVBQUUsSUFBSSxFQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQzNFLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxPQUFPLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsT0FBTyxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxRQUFRLEVBQUMsRUFBQyxDQUFDLENBQUM7UUFDNUUsT0FBTyxDQUFDLDZCQUE2QixDQUFDLE9BQU8sQ0FBQyxDQUFDO0tBQ2hEO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFpQjtJQUN4QyxVQUFVLEVBQUUsTUFBTTtJQUNsQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsTUFBMEI7Q0FDdkMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFVucGFjaywgVW5wYWNrQXR0cnMsIFVucGFja0lucHV0c30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuaW1wb3J0IHtzbGljZX0gZnJvbSAnLi9TbGljZSc7XG5cbmV4cG9ydCBmdW5jdGlvbiB1bnBhY2soXG4gICAgYXJnczoge2lucHV0czogVW5wYWNrSW5wdXRzLCBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSwgYXR0cnM6IFVucGFja0F0dHJzfSk6XG4gICAgVGVuc29ySW5mb1tdIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3ZhbHVlfSA9IGlucHV0cztcbiAgbGV0IHtheGlzfSA9IGF0dHJzO1xuXG4gIGlmIChheGlzIDwgMCkge1xuICAgIGF4aXMgKz0gdmFsdWUuc2hhcGUubGVuZ3RoO1xuICB9XG5cbiAgY29uc3QgdmFsdWVSYW5rID0gdmFsdWUuc2hhcGUubGVuZ3RoO1xuXG4gIGNvbnN0IG51bSA9IHZhbHVlLnNoYXBlW2F4aXNdO1xuICBjb25zdCBvdXRTaGFwZTogbnVtYmVyW10gPSBuZXcgQXJyYXkodmFsdWVSYW5rIC0gMSk7XG4gIGxldCBvdXRJbmRleCA9IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVSYW5rOyBpKyspIHtcbiAgICBpZiAoaSAhPT0gYXhpcykge1xuICAgICAgb3V0U2hhcGVbb3V0SW5kZXgrK10gPSB2YWx1ZS5zaGFwZVtpXTtcbiAgICB9XG4gIH1cblxuICBjb25zdCBiZWdpbiA9IG5ldyBBcnJheSh2YWx1ZVJhbmspLmZpbGwoMCk7XG4gIGNvbnN0IHNpemUgPSB2YWx1ZS5zaGFwZS5zbGljZSgpO1xuICBzaXplW2F4aXNdID0gMTtcbiAgY29uc3QgcmVzID0gbmV3IEFycmF5KG51bSk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmVzLmxlbmd0aDsgaSsrKSB7XG4gICAgYmVnaW5bYXhpc10gPSBpO1xuICAgIGNvbnN0IHRlbXBSZXMgPSBzbGljZSh7aW5wdXRzOiB7eDogdmFsdWV9LCBiYWNrZW5kLCBhdHRyczoge2JlZ2luLCBzaXplfX0pO1xuICAgIHJlc1tpXSA9IHJlc2hhcGUoe2lucHV0czoge3g6IHRlbXBSZXN9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiBvdXRTaGFwZX19KTtcbiAgICBiYWNrZW5kLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKHRlbXBSZXMpO1xuICB9XG5cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IHVucGFja0NvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBVbnBhY2ssXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogdW5wYWNrIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=