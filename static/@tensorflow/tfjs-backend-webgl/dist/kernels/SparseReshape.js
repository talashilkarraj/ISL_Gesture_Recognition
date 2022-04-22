/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { SparseReshape } from '@tensorflow/tfjs-core';
import { sparseReshapeImplCPU } from '../kernel_utils/shared';
export function sparseReshape(args) {
    const { inputs, backend } = args;
    const { inputIndices, inputShape, newShape } = inputs;
    if (inputIndices.shape.length !== 2) {
        throw new Error(`Input indices should be a matrix but received shape ${inputIndices.shape}`);
    }
    if (inputShape.shape.length !== 1) {
        throw new Error(`Input shape should be a vector but received shape ${inputShape.shape}`);
    }
    if (newShape.shape.length !== 1) {
        throw new Error(`Target shape should be a vector but received shape ${newShape.shape}`);
    }
    const $inputShape = Array.from(backend.readSync(inputShape.dataId));
    const $inputIndices = backend.readSync(inputIndices.dataId);
    const targetShape = Array.from(backend.readSync(newShape.dataId));
    const [newIndices, indicesShape, outputShape] = sparseReshapeImplCPU($inputIndices, inputIndices.shape, inputIndices.dtype, $inputShape, targetShape);
    return [
        backend.makeTensorInfo(indicesShape, inputIndices.dtype, newIndices),
        backend.makeTensorInfo([outputShape.length], newShape.dtype, new Int32Array(outputShape)),
    ];
}
export const sparseReshapeConfig = {
    kernelName: SparseReshape,
    backendName: 'webgl',
    kernelFunc: sparseReshape,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3BhcnNlUmVzaGFwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMva2VybmVscy9TcGFyc2VSZXNoYXBlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBZSxhQUFhLEVBQThDLE1BQU0sdUJBQXVCLENBQUM7QUFHL0csT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFNUQsTUFBTSxVQUFVLGFBQWEsQ0FDekIsSUFBOEQ7SUFFaEUsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDL0IsTUFBTSxFQUFDLFlBQVksRUFBRSxVQUFVLEVBQUUsUUFBUSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ3BELElBQUksWUFBWSxDQUFDLEtBQUssQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ25DLE1BQU0sSUFBSSxLQUFLLENBQUMsdURBQ1osWUFBWSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDM0I7SUFDRCxJQUFJLFVBQVUsQ0FBQyxLQUFLLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUNqQyxNQUFNLElBQUksS0FBSyxDQUFDLHFEQUNaLFVBQVUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQ3pCO0lBRUQsSUFBSSxRQUFRLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7UUFDL0IsTUFBTSxJQUFJLEtBQUssQ0FDWCxzREFBc0QsUUFBUSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDN0U7SUFFRCxNQUFNLFdBQVcsR0FDYixLQUFLLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxRQUFRLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBZSxDQUFDLENBQUM7SUFDbEUsTUFBTSxhQUFhLEdBQUcsT0FBTyxDQUFDLFFBQVEsQ0FBQyxZQUFZLENBQUMsTUFBTSxDQUFlLENBQUM7SUFDMUUsTUFBTSxXQUFXLEdBQ2IsS0FBSyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQWUsQ0FBQyxDQUFDO0lBRWhFLE1BQU0sQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLFdBQVcsQ0FBQyxHQUFHLG9CQUFvQixDQUNoRSxhQUFhLEVBQUUsWUFBWSxDQUFDLEtBQUssRUFBRSxZQUFZLENBQUMsS0FBSyxFQUFFLFdBQVcsRUFDbEUsV0FBVyxDQUFDLENBQUM7SUFDakIsT0FBTztRQUNMLE9BQU8sQ0FBQyxjQUFjLENBQUMsWUFBWSxFQUFFLFlBQVksQ0FBQyxLQUFLLEVBQUUsVUFBVSxDQUFDO1FBQ3BFLE9BQU8sQ0FBQyxjQUFjLENBQ2xCLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxFQUFFLFFBQVEsQ0FBQyxLQUFLLEVBQUUsSUFBSSxVQUFVLENBQUMsV0FBVyxDQUFDLENBQUM7S0FDdkUsQ0FBQztBQUNKLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxtQkFBbUIsR0FBaUI7SUFDL0MsVUFBVSxFQUFFLGFBQWE7SUFDekIsV0FBVyxFQUFFLE9BQU87SUFDcEIsVUFBVSxFQUFFLGFBQWE7Q0FDMUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIFNwYXJzZVJlc2hhcGUsIFNwYXJzZVJlc2hhcGVJbnB1dHMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXl9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRXZWJHTH0gZnJvbSAnLi4vYmFja2VuZF93ZWJnbCc7XG5pbXBvcnQge3NwYXJzZVJlc2hhcGVJbXBsQ1BVfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvc2hhcmVkJztcblxuZXhwb3J0IGZ1bmN0aW9uIHNwYXJzZVJlc2hhcGUoXG4gICAgYXJnczoge2lucHV0czogU3BhcnNlUmVzaGFwZUlucHV0cywgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTH0pOlxuICAgIFtUZW5zb3JJbmZvLCBUZW5zb3JJbmZvXSB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmR9ID0gYXJncztcbiAgY29uc3Qge2lucHV0SW5kaWNlcywgaW5wdXRTaGFwZSwgbmV3U2hhcGV9ID0gaW5wdXRzO1xuICBpZiAoaW5wdXRJbmRpY2VzLnNoYXBlLmxlbmd0aCAhPT0gMikge1xuICAgIHRocm93IG5ldyBFcnJvcihgSW5wdXQgaW5kaWNlcyBzaG91bGQgYmUgYSBtYXRyaXggYnV0IHJlY2VpdmVkIHNoYXBlICR7XG4gICAgICAgIGlucHV0SW5kaWNlcy5zaGFwZX1gKTtcbiAgfVxuICBpZiAoaW5wdXRTaGFwZS5zaGFwZS5sZW5ndGggIT09IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYElucHV0IHNoYXBlIHNob3VsZCBiZSBhIHZlY3RvciBidXQgcmVjZWl2ZWQgc2hhcGUgJHtcbiAgICAgICAgaW5wdXRTaGFwZS5zaGFwZX1gKTtcbiAgfVxuXG4gIGlmIChuZXdTaGFwZS5zaGFwZS5sZW5ndGggIT09IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBUYXJnZXQgc2hhcGUgc2hvdWxkIGJlIGEgdmVjdG9yIGJ1dCByZWNlaXZlZCBzaGFwZSAke25ld1NoYXBlLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3QgJGlucHV0U2hhcGUgPVxuICAgICAgQXJyYXkuZnJvbShiYWNrZW5kLnJlYWRTeW5jKGlucHV0U2hhcGUuZGF0YUlkKSBhcyBUeXBlZEFycmF5KTtcbiAgY29uc3QgJGlucHV0SW5kaWNlcyA9IGJhY2tlbmQucmVhZFN5bmMoaW5wdXRJbmRpY2VzLmRhdGFJZCkgYXMgVHlwZWRBcnJheTtcbiAgY29uc3QgdGFyZ2V0U2hhcGUgPVxuICAgICAgQXJyYXkuZnJvbShiYWNrZW5kLnJlYWRTeW5jKG5ld1NoYXBlLmRhdGFJZCkgYXMgVHlwZWRBcnJheSk7XG5cbiAgY29uc3QgW25ld0luZGljZXMsIGluZGljZXNTaGFwZSwgb3V0cHV0U2hhcGVdID0gc3BhcnNlUmVzaGFwZUltcGxDUFUoXG4gICAgICAkaW5wdXRJbmRpY2VzLCBpbnB1dEluZGljZXMuc2hhcGUsIGlucHV0SW5kaWNlcy5kdHlwZSwgJGlucHV0U2hhcGUsXG4gICAgICB0YXJnZXRTaGFwZSk7XG4gIHJldHVybiBbXG4gICAgYmFja2VuZC5tYWtlVGVuc29ySW5mbyhpbmRpY2VzU2hhcGUsIGlucHV0SW5kaWNlcy5kdHlwZSwgbmV3SW5kaWNlcyksXG4gICAgYmFja2VuZC5tYWtlVGVuc29ySW5mbyhcbiAgICAgICAgW291dHB1dFNoYXBlLmxlbmd0aF0sIG5ld1NoYXBlLmR0eXBlLCBuZXcgSW50MzJBcnJheShvdXRwdXRTaGFwZSkpLFxuICBdO1xufVxuXG5leHBvcnQgY29uc3Qgc3BhcnNlUmVzaGFwZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBTcGFyc2VSZXNoYXBlLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdsJyxcbiAga2VybmVsRnVuYzogc3BhcnNlUmVzaGFwZSxcbn07XG4iXX0=