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
import { ArgMin, backend_util, util } from '@tensorflow/tfjs-core';
import { argMinMaxReduce } from '../kernel_utils/arg_min_max';
import { transpose } from './Transpose';
export function argMin(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    let axes = util.parseAxisParam(axis, x.shape);
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('argMin', [axes[0]], $x.shape.length);
    const out = argMinMaxReduce(backend, $x, axes[0], 'min');
    intermediateTensorInfos.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return out;
}
export const argMinConfig = {
    kernelName: ArgMin,
    backendName: 'webgl',
    kernelFunc: argMin
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXJnTWluLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL0FyZ01pbi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUE2QixZQUFZLEVBQXdDLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBR2xJLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSw2QkFBNkIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXRDLE1BQU0sVUFBVSxNQUFNLENBQ2xCLElBQ3lFO0lBRTNFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25CLE1BQU0sRUFBQyxJQUFJLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFckIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzlDLE1BQU0sWUFBWSxHQUFHLFlBQVksQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUMzRSxJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDWCxNQUFNLHVCQUF1QixHQUFHLEVBQUUsQ0FBQztJQUNuQyxJQUFJLFlBQVksSUFBSSxJQUFJLEVBQUU7UUFDeEIsRUFBRSxHQUFHLFNBQVMsQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUUsWUFBWSxFQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQ3BFLHVCQUF1QixDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUNqQyxJQUFJLEdBQUcsWUFBWSxDQUFDLGdCQUFnQixDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztLQUNwRTtJQUVELFlBQVksQ0FBQywwQkFBMEIsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBRTlFLE1BQU0sR0FBRyxHQUFHLGVBQWUsQ0FBQyxPQUFPLEVBQUUsRUFBRSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUV6RCx1QkFBdUIsQ0FBQyxPQUFPLENBQzNCLENBQUMsQ0FBQyxFQUFFLENBQUMsT0FBTyxDQUFDLDZCQUE2QixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDbkQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFpQjtJQUN4QyxVQUFVLEVBQUUsTUFBTTtJQUNsQixXQUFXLEVBQUUsT0FBTztJQUNwQixVQUFVLEVBQUUsTUFBMEI7Q0FDdkMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtBcmdNaW4sIEFyZ01pbkF0dHJzLCBBcmdNaW5JbnB1dHMsIGJhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBUZW5zb3JJbmZvLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge01hdGhCYWNrZW5kV2ViR0x9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ2wnO1xuaW1wb3J0IHthcmdNaW5NYXhSZWR1Y2V9IGZyb20gJy4uL2tlcm5lbF91dGlscy9hcmdfbWluX21heCc7XG5pbXBvcnQge3RyYW5zcG9zZX0gZnJvbSAnLi9UcmFuc3Bvc2UnO1xuXG5leHBvcnQgZnVuY3Rpb24gYXJnTWluKFxuICAgIGFyZ3M6XG4gICAgICAgIHtpbnB1dHM6IEFyZ01pbklucHV0cywgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTCwgYXR0cnM6IEFyZ01pbkF0dHJzfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHt4fSA9IGlucHV0cztcbiAgY29uc3Qge2F4aXN9ID0gYXR0cnM7XG5cbiAgbGV0IGF4ZXMgPSB1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHguc2hhcGUpO1xuICBjb25zdCBwZXJtdXRlZEF4ZXMgPSBiYWNrZW5kX3V0aWwuZ2V0QXhlc1Blcm11dGF0aW9uKGF4ZXMsIHguc2hhcGUubGVuZ3RoKTtcbiAgbGV0ICR4ID0geDtcbiAgY29uc3QgaW50ZXJtZWRpYXRlVGVuc29ySW5mb3MgPSBbXTtcbiAgaWYgKHBlcm11dGVkQXhlcyAhPSBudWxsKSB7XG4gICAgJHggPSB0cmFuc3Bvc2Uoe2lucHV0czoge3h9LCBiYWNrZW5kLCBhdHRyczoge3Blcm06IHBlcm11dGVkQXhlc319KTtcbiAgICBpbnRlcm1lZGlhdGVUZW5zb3JJbmZvcy5wdXNoKCR4KTtcbiAgICBheGVzID0gYmFja2VuZF91dGlsLmdldElubmVyTW9zdEF4ZXMoYXhlcy5sZW5ndGgsICR4LnNoYXBlLmxlbmd0aCk7XG4gIH1cblxuICBiYWNrZW5kX3V0aWwuYXNzZXJ0QXhlc0FyZUlubmVyTW9zdERpbXMoJ2FyZ01pbicsIFtheGVzWzBdXSwgJHguc2hhcGUubGVuZ3RoKTtcblxuICBjb25zdCBvdXQgPSBhcmdNaW5NYXhSZWR1Y2UoYmFja2VuZCwgJHgsIGF4ZXNbMF0sICdtaW4nKTtcblxuICBpbnRlcm1lZGlhdGVUZW5zb3JJbmZvcy5mb3JFYWNoKFxuICAgICAgdCA9PiBiYWNrZW5kLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKHQpKTtcbiAgcmV0dXJuIG91dDtcbn1cblxuZXhwb3J0IGNvbnN0IGFyZ01pbkNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBBcmdNaW4sXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBhcmdNaW4gYXMge30gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==