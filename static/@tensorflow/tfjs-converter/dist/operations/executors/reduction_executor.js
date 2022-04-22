/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';
import { getParamValue } from './utils';
export const executeOp = (node, tensorMap, context) => {
    switch (node.op) {
        case 'Max': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.max(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'Mean': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.mean(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'Min': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.min(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'Sum': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.sum(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'All': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.all(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'Any': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.any(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'ArgMax': {
            const axis = getParamValue('axis', node, tensorMap, context);
            return [tfOps.argMax(getParamValue('x', node, tensorMap, context), axis)];
        }
        case 'ArgMin': {
            const axis = getParamValue('axis', node, tensorMap, context);
            return [tfOps.argMin(getParamValue('x', node, tensorMap, context), axis)];
        }
        case 'Prod': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const keepDims = getParamValue('keepDims', node, tensorMap, context);
            return [tfOps.prod(getParamValue('x', node, tensorMap, context), axis, keepDims)];
        }
        case 'Cumprod': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const exclusive = getParamValue('exclusive', node, tensorMap, context);
            const reverse = getParamValue('reverse', node, tensorMap, context);
            return [tfOps.cumprod(getParamValue('x', node, tensorMap, context), axis, exclusive, reverse)];
        }
        case 'Cumsum': {
            const axis = getParamValue('axis', node, tensorMap, context);
            const exclusive = getParamValue('exclusive', node, tensorMap, context);
            const reverse = getParamValue('reverse', node, tensorMap, context);
            return [tfOps.cumsum(getParamValue('x', node, tensorMap, context), axis, exclusive, reverse)];
        }
        case 'Bincount':
            const x = getParamValue('x', node, tensorMap, context);
            const weights = getParamValue('weights', node, tensorMap, context);
            const size = getParamValue('size', node, tensorMap, context);
            return [tfOps.bincount(x, weights, size)];
        case 'DenseBincount': {
            const x = getParamValue('x', node, tensorMap, context);
            const weights = getParamValue('weights', node, tensorMap, context);
            const size = getParamValue('size', node, tensorMap, context);
            const binaryOutput = getParamValue('binaryOutput', node, tensorMap, context);
            return [tfOps.denseBincount(x, weights, size, binaryOutput)];
        }
        default:
            throw TypeError(`Node type ${node.op} is not implemented`);
    }
};
export const CATEGORY = 'reduction';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVkdWN0aW9uX2V4ZWN1dG9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb252ZXJ0ZXIvc3JjL29wZXJhdGlvbnMvZXhlY3V0b3JzL3JlZHVjdGlvbl9leGVjdXRvci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxpREFBaUQ7QUFDakQsT0FBTyxLQUFLLEtBQUssTUFBTSxrREFBa0QsQ0FBQztBQU0xRSxPQUFPLEVBQUMsYUFBYSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRXRDLE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FDbEIsQ0FBQyxJQUFVLEVBQUUsU0FBMEIsRUFDdEMsT0FBeUIsRUFBWSxFQUFFO0lBQ3RDLFFBQVEsSUFBSSxDQUFDLEVBQUUsRUFBRTtRQUNmLEtBQUssS0FBSyxDQUFDLENBQUM7WUFDVixNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFhLENBQUM7WUFDaEUsTUFBTSxRQUFRLEdBQ1YsYUFBYSxDQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBWSxDQUFDO1lBQ25FLE9BQU8sQ0FBQyxLQUFLLENBQUMsR0FBRyxDQUNiLGFBQWEsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsRUFBRSxJQUFJLEVBQzVELFFBQVEsQ0FBQyxDQUFDLENBQUM7U0FDaEI7UUFDRCxLQUFLLE1BQU0sQ0FBQyxDQUFDO1lBQ1gsTUFBTSxJQUFJLEdBQ04sYUFBYSxDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ2hFLE1BQU0sUUFBUSxHQUNWLGFBQWEsQ0FBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVksQ0FBQztZQUNuRSxPQUFPLENBQUMsS0FBSyxDQUFDLElBQUksQ0FDZCxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLEVBQUUsSUFBSSxFQUM1RCxRQUFRLENBQUMsQ0FBQyxDQUFDO1NBQ2hCO1FBQ0QsS0FBSyxLQUFLLENBQUMsQ0FBQztZQUNWLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUNoRSxNQUFNLFFBQVEsR0FDVixhQUFhLENBQUMsVUFBVSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFZLENBQUM7WUFDbkUsT0FBTyxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQ2IsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxFQUFFLElBQUksRUFDNUQsUUFBUSxDQUFDLENBQUMsQ0FBQztTQUNoQjtRQUNELEtBQUssS0FBSyxDQUFDLENBQUM7WUFDVixNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFhLENBQUM7WUFDaEUsTUFBTSxRQUFRLEdBQ1YsYUFBYSxDQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBWSxDQUFDO1lBQ25FLE9BQU8sQ0FBQyxLQUFLLENBQUMsR0FBRyxDQUNiLGFBQWEsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsRUFBRSxJQUFJLEVBQzVELFFBQVEsQ0FBQyxDQUFDLENBQUM7U0FDaEI7UUFDRCxLQUFLLEtBQUssQ0FBQyxDQUFDO1lBQ1YsTUFBTSxJQUFJLEdBQ04sYUFBYSxDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ2hFLE1BQU0sUUFBUSxHQUNWLGFBQWEsQ0FBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVksQ0FBQztZQUNuRSxPQUFPLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FDYixhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLEVBQUUsSUFBSSxFQUM1RCxRQUFRLENBQUMsQ0FBQyxDQUFDO1NBQ2hCO1FBQ0QsS0FBSyxLQUFLLENBQUMsQ0FBQztZQUNWLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUNoRSxNQUFNLFFBQVEsR0FDVixhQUFhLENBQUMsVUFBVSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFZLENBQUM7WUFDbkUsT0FBTyxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQ2IsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxFQUFFLElBQUksRUFDNUQsUUFBUSxDQUFDLENBQUMsQ0FBQztTQUNoQjtRQUNELEtBQUssUUFBUSxDQUFDLENBQUM7WUFDYixNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsT0FBTyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQ2hCLGFBQWEsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO1NBQ3BFO1FBQ0QsS0FBSyxRQUFRLENBQUMsQ0FBQztZQUNiLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUM5RCxPQUFPLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FDaEIsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7U0FDcEU7UUFDRCxLQUFLLE1BQU0sQ0FBQyxDQUFDO1lBQ1gsTUFBTSxJQUFJLEdBQ04sYUFBYSxDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ2hFLE1BQU0sUUFBUSxHQUNWLGFBQWEsQ0FBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVksQ0FBQztZQUNuRSxPQUFPLENBQUMsS0FBSyxDQUFDLElBQUksQ0FDZCxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLEVBQUUsSUFBSSxFQUM1RCxRQUFRLENBQUMsQ0FBQyxDQUFDO1NBQ2hCO1FBQ0QsS0FBSyxTQUFTLENBQUMsQ0FBQztZQUNkLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUM5RCxNQUFNLFNBQVMsR0FDWCxhQUFhLENBQUMsV0FBVyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFZLENBQUM7WUFDcEUsTUFBTSxPQUFPLEdBQ1QsYUFBYSxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBWSxDQUFDO1lBQ2xFLE9BQU8sQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUNqQixhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLEVBQUUsSUFBSSxFQUM1RCxTQUFTLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztTQUMxQjtRQUNELEtBQUssUUFBUSxDQUFDLENBQUM7WUFDYixNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsTUFBTSxTQUFTLEdBQ1gsYUFBYSxDQUFDLFdBQVcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBWSxDQUFDO1lBQ3BFLE1BQU0sT0FBTyxHQUNULGFBQWEsQ0FBQyxTQUFTLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVksQ0FBQztZQUNsRSxPQUFPLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FDaEIsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxFQUFFLElBQUksRUFDNUQsU0FBUyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7U0FDMUI7UUFDRCxLQUFLLFVBQVU7WUFDYixNQUFNLENBQUMsR0FBRyxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFhLENBQUM7WUFDbkUsTUFBTSxPQUFPLEdBQ1QsYUFBYSxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ25FLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUU5RCxPQUFPLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLEVBQUUsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7UUFDNUMsS0FBSyxlQUFlLENBQUMsQ0FBQztZQUNwQixNQUFNLENBQUMsR0FBRyxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUN6QyxDQUFDO1lBQ2IsTUFBTSxPQUFPLEdBQ1QsYUFBYSxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FDekMsQ0FBQztZQUNiLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUU5RCxNQUFNLFlBQVksR0FDZCxhQUFhLENBQUMsY0FBYyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUMvQyxDQUFDO1lBRVosT0FBTyxDQUFDLEtBQUssQ0FBQyxhQUFhLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLEVBQUUsWUFBWSxDQUFDLENBQUMsQ0FBQztTQUM5RDtRQUNEO1lBQ0UsTUFBTSxTQUFTLENBQUMsYUFBYSxJQUFJLENBQUMsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0tBQzlEO0FBQ0gsQ0FBQyxDQUFDO0FBRU4sTUFBTSxDQUFDLE1BQU0sUUFBUSxHQUFHLFdBQVcsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8taW1wb3J0cy1mcm9tLWRpc3RcbmltcG9ydCAqIGFzIHRmT3BzIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZS9kaXN0L29wcy9vcHNfZm9yX2NvbnZlcnRlcic7XG5cbmltcG9ydCB7TmFtZWRUZW5zb3JzTWFwfSBmcm9tICcuLi8uLi9kYXRhL3R5cGVzJztcbmltcG9ydCB7RXhlY3V0aW9uQ29udGV4dH0gZnJvbSAnLi4vLi4vZXhlY3V0b3IvZXhlY3V0aW9uX2NvbnRleHQnO1xuaW1wb3J0IHtJbnRlcm5hbE9wRXhlY3V0b3IsIE5vZGV9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtnZXRQYXJhbVZhbHVlfSBmcm9tICcuL3V0aWxzJztcblxuZXhwb3J0IGNvbnN0IGV4ZWN1dGVPcDogSW50ZXJuYWxPcEV4ZWN1dG9yID1cbiAgICAobm9kZTogTm9kZSwgdGVuc29yTWFwOiBOYW1lZFRlbnNvcnNNYXAsXG4gICAgIGNvbnRleHQ6IEV4ZWN1dGlvbkNvbnRleHQpOiBUZW5zb3JbXSA9PiB7XG4gICAgICBzd2l0Y2ggKG5vZGUub3ApIHtcbiAgICAgICAgY2FzZSAnTWF4Jzoge1xuICAgICAgICAgIGNvbnN0IGF4aXMgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdheGlzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgICAgICBjb25zdCBrZWVwRGltcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2tlZXBEaW1zJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBib29sZWFuO1xuICAgICAgICAgIHJldHVybiBbdGZPcHMubWF4KFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IsIGF4aXMsXG4gICAgICAgICAgICAgIGtlZXBEaW1zKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnTWVhbic6IHtcbiAgICAgICAgICBjb25zdCBheGlzID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnYXhpcycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG4gICAgICAgICAgY29uc3Qga2VlcERpbXMgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdrZWVwRGltcycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYm9vbGVhbjtcbiAgICAgICAgICByZXR1cm4gW3RmT3BzLm1lYW4oXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAga2VlcERpbXMpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdNaW4nOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgICAgIGNvbnN0IGtlZXBEaW1zID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgna2VlcERpbXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIGJvb2xlYW47XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5taW4oXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAga2VlcERpbXMpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdTdW0nOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgICAgIGNvbnN0IGtlZXBEaW1zID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgna2VlcERpbXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIGJvb2xlYW47XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5zdW0oXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAga2VlcERpbXMpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdBbGwnOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgICAgIGNvbnN0IGtlZXBEaW1zID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgna2VlcERpbXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIGJvb2xlYW47XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5hbGwoXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAga2VlcERpbXMpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdBbnknOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgICAgIGNvbnN0IGtlZXBEaW1zID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgna2VlcERpbXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIGJvb2xlYW47XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5hbnkoXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAga2VlcERpbXMpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdBcmdNYXgnOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgICAgICByZXR1cm4gW3RmT3BzLmFyZ01heChcbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yLCBheGlzKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnQXJnTWluJzoge1xuICAgICAgICAgIGNvbnN0IGF4aXMgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdheGlzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5hcmdNaW4oXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyldO1xuICAgICAgICB9XG4gICAgICAgIGNhc2UgJ1Byb2QnOiB7XG4gICAgICAgICAgY29uc3QgYXhpcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgICAgIGNvbnN0IGtlZXBEaW1zID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgna2VlcERpbXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIGJvb2xlYW47XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5wcm9kKFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IsIGF4aXMsXG4gICAgICAgICAgICAgIGtlZXBEaW1zKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnQ3VtcHJvZCc6IHtcbiAgICAgICAgICBjb25zdCBheGlzID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnYXhpcycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IGV4Y2x1c2l2ZSA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2V4Y2x1c2l2ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYm9vbGVhbjtcbiAgICAgICAgICBjb25zdCByZXZlcnNlID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgncmV2ZXJzZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYm9vbGVhbjtcbiAgICAgICAgICByZXR1cm4gW3RmT3BzLmN1bXByb2QoXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvciwgYXhpcyxcbiAgICAgICAgICAgICAgZXhjbHVzaXZlLCByZXZlcnNlKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnQ3Vtc3VtJzoge1xuICAgICAgICAgIGNvbnN0IGF4aXMgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdheGlzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgY29uc3QgZXhjbHVzaXZlID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZXhjbHVzaXZlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBib29sZWFuO1xuICAgICAgICAgIGNvbnN0IHJldmVyc2UgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdyZXZlcnNlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBib29sZWFuO1xuICAgICAgICAgIHJldHVybiBbdGZPcHMuY3Vtc3VtKFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IsIGF4aXMsXG4gICAgICAgICAgICAgIGV4Y2x1c2l2ZSwgcmV2ZXJzZSldO1xuICAgICAgICB9XG4gICAgICAgIGNhc2UgJ0JpbmNvdW50JzpcbiAgICAgICAgICBjb25zdCB4ID0gZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yMUQ7XG4gICAgICAgICAgY29uc3Qgd2VpZ2h0cyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3dlaWdodHMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjFEO1xuICAgICAgICAgIGNvbnN0IHNpemUgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzaXplJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG5cbiAgICAgICAgICByZXR1cm4gW3RmT3BzLmJpbmNvdW50KHgsIHdlaWdodHMsIHNpemUpXTtcbiAgICAgICAgY2FzZSAnRGVuc2VCaW5jb3VudCc6IHtcbiAgICAgICAgICBjb25zdCB4ID0gZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yMUQgfFxuICAgICAgICAgICAgICBUZW5zb3IyRDtcbiAgICAgICAgICBjb25zdCB3ZWlnaHRzID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnd2VpZ2h0cycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yMUQgfFxuICAgICAgICAgICAgICBUZW5zb3IyRDtcbiAgICAgICAgICBjb25zdCBzaXplID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnc2l6ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuXG4gICAgICAgICAgY29uc3QgYmluYXJ5T3V0cHV0ID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnYmluYXJ5T3V0cHV0Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhc1xuICAgICAgICAgICAgICBib29sZWFuO1xuXG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5kZW5zZUJpbmNvdW50KHgsIHdlaWdodHMsIHNpemUsIGJpbmFyeU91dHB1dCldO1xuICAgICAgICB9XG4gICAgICAgIGRlZmF1bHQ6XG4gICAgICAgICAgdGhyb3cgVHlwZUVycm9yKGBOb2RlIHR5cGUgJHtub2RlLm9wfSBpcyBub3QgaW1wbGVtZW50ZWRgKTtcbiAgICAgIH1cbiAgICB9O1xuXG5leHBvcnQgY29uc3QgQ0FURUdPUlkgPSAncmVkdWN0aW9uJztcbiJdfQ==