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
import { AvgPool3DGrad, backend_util } from '@tensorflow/tfjs-core';
import { AvgPool3DBackpropProgram } from '../avg_pool_backprop_gpu';
export function avgPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = [1, 1, 1];
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    const avgPoolBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
    return backend.runWebGLProgram(avgPoolBackpropProgram, [dy], x.dtype);
}
export const avgPool3DGradConfig = {
    kernelName: AvgPool3DGrad,
    backendName: 'webgl',
    kernelFunc: avgPool3DGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXZnUG9vbDNER3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMva2VybmVscy9BdmdQb29sM0RHcmFkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxhQUFhLEVBQTJDLFlBQVksRUFBdUMsTUFBTSx1QkFBdUIsQ0FBQztBQUVqSixPQUFPLEVBQUMsd0JBQXdCLEVBQUMsTUFBTSwwQkFBMEIsQ0FBQztBQUdsRSxNQUFNLFVBQVUsYUFBYSxDQUFDLElBSTdCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxFQUFFLEVBQUUsS0FBSyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQzNCLE1BQU0sQ0FBQyxHQUFHLEtBQUssQ0FBQztJQUNoQixNQUFNLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBQzFELE1BQU0sU0FBUyxHQUE2QixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFdEQsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLGlCQUFpQixDQUMzQyxDQUFDLENBQUMsS0FBaUQsRUFBRSxVQUFVLEVBQUUsT0FBTyxFQUN4RSxTQUFTLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQ3JDLE1BQU0sc0JBQXNCLEdBQUcsSUFBSSx3QkFBd0IsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUN0RSxPQUFPLE9BQU8sQ0FBQyxlQUFlLENBQUMsc0JBQXNCLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDeEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLG1CQUFtQixHQUFpQjtJQUMvQyxVQUFVLEVBQUUsYUFBYTtJQUN6QixXQUFXLEVBQUUsT0FBTztJQUNwQixVQUFVLEVBQUUsYUFBaUM7Q0FDOUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7QXZnUG9vbDNER3JhZCwgQXZnUG9vbDNER3JhZEF0dHJzLCBBdmdQb29sM0RHcmFkSW5wdXRzLCBiYWNrZW5kX3V0aWwsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtBdmdQb29sM0RCYWNrcHJvcFByb2dyYW19IGZyb20gJy4uL2F2Z19wb29sX2JhY2twcm9wX2dwdSc7XG5pbXBvcnQge01hdGhCYWNrZW5kV2ViR0x9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ2wnO1xuXG5leHBvcnQgZnVuY3Rpb24gYXZnUG9vbDNER3JhZChhcmdzOiB7XG4gIGlucHV0czogQXZnUG9vbDNER3JhZElucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTCxcbiAgYXR0cnM6IEF2Z1Bvb2wzREdyYWRBdHRyc1xufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7ZHksIGlucHV0fSA9IGlucHV0cztcbiAgY29uc3QgeCA9IGlucHV0O1xuICBjb25zdCB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkLCBkaW1Sb3VuZGluZ01vZGV9ID0gYXR0cnM7XG4gIGNvbnN0IGRpbGF0aW9uczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEsIDEsIDFdO1xuXG4gIGNvbnN0IGNvbnZJbmZvID0gYmFja2VuZF91dGlsLmNvbXB1dGVQb29sM0RJbmZvKFxuICAgICAgeC5zaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmaWx0ZXJTaXplLCBzdHJpZGVzLFxuICAgICAgZGlsYXRpb25zLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IGF2Z1Bvb2xCYWNrcHJvcFByb2dyYW0gPSBuZXcgQXZnUG9vbDNEQmFja3Byb3BQcm9ncmFtKGNvbnZJbmZvKTtcbiAgcmV0dXJuIGJhY2tlbmQucnVuV2ViR0xQcm9ncmFtKGF2Z1Bvb2xCYWNrcHJvcFByb2dyYW0sIFtkeV0sIHguZHR5cGUpO1xufVxuXG5leHBvcnQgY29uc3QgYXZnUG9vbDNER3JhZENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBBdmdQb29sM0RHcmFkLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdsJyxcbiAga2VybmVsRnVuYzogYXZnUG9vbDNER3JhZCBhcyB7fSBhcyBLZXJuZWxGdW5jXG59O1xuIl19