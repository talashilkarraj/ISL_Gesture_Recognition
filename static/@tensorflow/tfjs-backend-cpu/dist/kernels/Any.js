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
import { Any, backend_util, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { reshape } from './Reshape';
import { transpose } from './Transpose';
export function any(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'any');
    const origAxes = util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        axes = backend_util.getInnerMostAxes(axes.length, x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('any', axes, $x.shape.length);
    const [outShape, reduceShape] = backend_util.computeOutAndReduceShapes($x.shape, axes);
    const reduceSize = util.sizeFromShape(reduceShape);
    const vals = util.makeZerosTypedArray(util.sizeFromShape(outShape), $x.dtype);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let anyVal = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            anyVal = anyVal || value;
        }
        vals[i] = anyVal;
    }
    if (permutedAxes != null) {
        backend.disposeIntermediateTensorInfo($x);
    }
    const result = backend.makeTensorInfo(outShape, $x.dtype, vals);
    if (keepDims) {
        const expandedShape = backend_util.expandShapeToKeepDim(outShape, origAxes);
        const reshapedResult = reshape({ inputs: { x: result }, backend, attrs: { shape: expandedShape } });
        backend.disposeIntermediateTensorInfo(result);
        return reshapedResult;
    }
    return result;
}
export const anyConfig = {
    kernelName: Any,
    backendName: 'cpu',
    kernelFunc: any
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQW55LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9BbnkudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLEdBQUcsRUFBdUIsWUFBWSxFQUFvRCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUdySSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDN0MsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNsQyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXRDLE1BQU0sVUFBVSxHQUFHLENBQ2YsSUFBbUU7SUFFckUsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLElBQUksRUFBRSxRQUFRLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFL0IsZ0JBQWdCLENBQUMsQ0FBQyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBRTNCLE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUNwRCxJQUFJLElBQUksR0FBRyxRQUFRLENBQUM7SUFDcEIsTUFBTSxZQUFZLEdBQUcsWUFBWSxDQUFDLGtCQUFrQixDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQzNFLElBQUksRUFBRSxHQUFHLENBQUMsQ0FBQztJQUNYLElBQUksWUFBWSxJQUFJLElBQUksRUFBRTtRQUN4QixFQUFFLEdBQUcsU0FBUyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLElBQUksRUFBRSxZQUFZLEVBQUMsRUFBQyxDQUFDLENBQUM7UUFDcEUsSUFBSSxHQUFHLFlBQVksQ0FBQyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7S0FDbkU7SUFFRCxZQUFZLENBQUMsMEJBQTBCLENBQUMsS0FBSyxFQUFFLElBQUksRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ3RFLE1BQU0sQ0FBQyxRQUFRLEVBQUUsV0FBVyxDQUFDLEdBQ3pCLFlBQVksQ0FBQyx5QkFBeUIsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzNELE1BQU0sVUFBVSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDbkQsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDLG1CQUFtQixDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRTlFLE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxNQUFvQixDQUFDO0lBQy9ELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1FBQ3BDLE1BQU0sTUFBTSxHQUFHLENBQUMsR0FBRyxVQUFVLENBQUM7UUFDOUIsSUFBSSxNQUFNLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzNCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxVQUFVLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDbkMsTUFBTSxLQUFLLEdBQUcsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztZQUNoQyxNQUFNLEdBQUcsTUFBTSxJQUFJLEtBQUssQ0FBQztTQUMxQjtRQUNELElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUM7S0FDbEI7SUFFRCxJQUFJLFlBQVksSUFBSSxJQUFJLEVBQUU7UUFDeEIsT0FBTyxDQUFDLDZCQUE2QixDQUFDLEVBQUUsQ0FBQyxDQUFDO0tBQzNDO0lBRUQsTUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLGNBQWMsQ0FBQyxRQUFRLEVBQUUsRUFBRSxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztJQUVoRSxJQUFJLFFBQVEsRUFBRTtRQUNaLE1BQU0sYUFBYSxHQUFHLFlBQVksQ0FBQyxvQkFBb0IsQ0FBQyxRQUFRLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDNUUsTUFBTSxjQUFjLEdBQ2hCLE9BQU8sQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxNQUFNLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLGFBQWEsRUFBQyxFQUFDLENBQUMsQ0FBQztRQUUzRSxPQUFPLENBQUMsNkJBQTZCLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFOUMsT0FBTyxjQUFjLENBQUM7S0FDdkI7SUFFRCxPQUFPLE1BQU0sQ0FBQztBQUNoQixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFpQjtJQUNyQyxVQUFVLEVBQUUsR0FBRztJQUNmLFdBQVcsRUFBRSxLQUFLO0lBQ2xCLFVBQVUsRUFBRSxHQUF1QjtDQUNwQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0FueSwgQW55QXR0cnMsIEFueUlucHV0cywgYmFja2VuZF91dGlsLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7YXNzZXJ0Tm90Q29tcGxleH0gZnJvbSAnLi4vY3B1X3V0aWwnO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuaW1wb3J0IHt0cmFuc3Bvc2V9IGZyb20gJy4vVHJhbnNwb3NlJztcblxuZXhwb3J0IGZ1bmN0aW9uIGFueShcbiAgICBhcmdzOiB7aW5wdXRzOiBBbnlJbnB1dHMsIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLCBhdHRyczogQW55QXR0cnN9KTpcbiAgICBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7YXhpcywga2VlcERpbXN9ID0gYXR0cnM7XG5cbiAgYXNzZXJ0Tm90Q29tcGxleCh4LCAnYW55Jyk7XG5cbiAgY29uc3Qgb3JpZ0F4ZXMgPSB1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHguc2hhcGUpO1xuICBsZXQgYXhlcyA9IG9yaWdBeGVzO1xuICBjb25zdCBwZXJtdXRlZEF4ZXMgPSBiYWNrZW5kX3V0aWwuZ2V0QXhlc1Blcm11dGF0aW9uKGF4ZXMsIHguc2hhcGUubGVuZ3RoKTtcbiAgbGV0ICR4ID0geDtcbiAgaWYgKHBlcm11dGVkQXhlcyAhPSBudWxsKSB7XG4gICAgJHggPSB0cmFuc3Bvc2Uoe2lucHV0czoge3h9LCBiYWNrZW5kLCBhdHRyczoge3Blcm06IHBlcm11dGVkQXhlc319KTtcbiAgICBheGVzID0gYmFja2VuZF91dGlsLmdldElubmVyTW9zdEF4ZXMoYXhlcy5sZW5ndGgsIHguc2hhcGUubGVuZ3RoKTtcbiAgfVxuXG4gIGJhY2tlbmRfdXRpbC5hc3NlcnRBeGVzQXJlSW5uZXJNb3N0RGltcygnYW55JywgYXhlcywgJHguc2hhcGUubGVuZ3RoKTtcbiAgY29uc3QgW291dFNoYXBlLCByZWR1Y2VTaGFwZV0gPVxuICAgICAgYmFja2VuZF91dGlsLmNvbXB1dGVPdXRBbmRSZWR1Y2VTaGFwZXMoJHguc2hhcGUsIGF4ZXMpO1xuICBjb25zdCByZWR1Y2VTaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHJlZHVjZVNoYXBlKTtcbiAgY29uc3QgdmFscyA9IHV0aWwubWFrZVplcm9zVHlwZWRBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUob3V0U2hhcGUpLCAkeC5kdHlwZSk7XG5cbiAgY29uc3QgYVZhbHMgPSBiYWNrZW5kLmRhdGEuZ2V0KCR4LmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgdmFscy5sZW5ndGg7ICsraSkge1xuICAgIGNvbnN0IG9mZnNldCA9IGkgKiByZWR1Y2VTaXplO1xuICAgIGxldCBhbnlWYWwgPSBhVmFsc1tvZmZzZXRdO1xuICAgIGZvciAobGV0IGogPSAwOyBqIDwgcmVkdWNlU2l6ZTsgKytqKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IGFWYWxzW29mZnNldCArIGpdO1xuICAgICAgYW55VmFsID0gYW55VmFsIHx8IHZhbHVlO1xuICAgIH1cbiAgICB2YWxzW2ldID0gYW55VmFsO1xuICB9XG5cbiAgaWYgKHBlcm11dGVkQXhlcyAhPSBudWxsKSB7XG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbygkeCk7XG4gIH1cblxuICBjb25zdCByZXN1bHQgPSBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKG91dFNoYXBlLCAkeC5kdHlwZSwgdmFscyk7XG5cbiAgaWYgKGtlZXBEaW1zKSB7XG4gICAgY29uc3QgZXhwYW5kZWRTaGFwZSA9IGJhY2tlbmRfdXRpbC5leHBhbmRTaGFwZVRvS2VlcERpbShvdXRTaGFwZSwgb3JpZ0F4ZXMpO1xuICAgIGNvbnN0IHJlc2hhcGVkUmVzdWx0ID1cbiAgICAgICAgcmVzaGFwZSh7aW5wdXRzOiB7eDogcmVzdWx0fSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogZXhwYW5kZWRTaGFwZX19KTtcblxuICAgIGJhY2tlbmQuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ocmVzdWx0KTtcblxuICAgIHJldHVybiByZXNoYXBlZFJlc3VsdDtcbiAgfVxuXG4gIHJldHVybiByZXN1bHQ7XG59XG5cbmV4cG9ydCBjb25zdCBhbnlDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQW55LFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IGFueSBhcyB7fSBhcyBLZXJuZWxGdW5jXG59O1xuIl19