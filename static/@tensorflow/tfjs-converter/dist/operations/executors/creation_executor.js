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
        case 'Fill': {
            const shape = getParamValue('shape', node, tensorMap, context);
            const dtype = getParamValue('dtype', node, tensorMap, context);
            const value = getParamValue('value', node, tensorMap, context);
            return [tfOps.fill(shape, value, dtype)];
        }
        case 'LinSpace': {
            const start = getParamValue('start', node, tensorMap, context);
            const stop = getParamValue('stop', node, tensorMap, context);
            const num = getParamValue('num', node, tensorMap, context);
            return [tfOps.linspace(start, stop, num)];
        }
        case 'Multinomial': {
            const logits = getParamValue('logits', node, tensorMap, context);
            const numSamples = getParamValue('numSamples', node, tensorMap, context);
            const seed = getParamValue('seed', node, tensorMap, context);
            return [tfOps.multinomial(logits, numSamples, seed)];
        }
        case 'OneHot': {
            const indices = getParamValue('indices', node, tensorMap, context);
            const depth = getParamValue('depth', node, tensorMap, context);
            const onValue = getParamValue('onValue', node, tensorMap, context);
            const offValue = getParamValue('offValue', node, tensorMap, context);
            return [tfOps.oneHot(indices, depth, onValue, offValue)];
        }
        case 'Ones': {
            return [tfOps.ones(getParamValue('shape', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
        }
        case 'OnesLike': {
            return [tfOps.onesLike(getParamValue('x', node, tensorMap, context))];
        }
        case 'RandomUniform': {
            return [tfOps.randomUniform(
                // tslint:disable-next-line:no-any
                getParamValue('shape', node, tensorMap, context), getParamValue('minval', node, tensorMap, context), getParamValue('maxval', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
        }
        case 'Range': {
            const start = getParamValue('start', node, tensorMap, context);
            const stop = getParamValue('stop', node, tensorMap, context);
            const step = getParamValue('step', node, tensorMap, context);
            return [tfOps.range(start, stop, step, getParamValue('dtype', node, tensorMap, context))];
        }
        case 'TruncatedNormal': {
            const shape = getParamValue('shape', node, tensorMap, context);
            const mean = getParamValue('mean', node, tensorMap, context);
            const stdDev = getParamValue('stdDev', node, tensorMap, context);
            const seed = getParamValue('seed', node, tensorMap, context);
            return [tfOps.truncatedNormal(shape, mean, stdDev, getParamValue('dtype', node, tensorMap, context), seed)];
        }
        case 'Zeros': {
            return [tfOps.zeros(getParamValue('shape', node, tensorMap, context), getParamValue('dtype', node, tensorMap, context))];
        }
        case 'ZerosLike': {
            return [tfOps.zerosLike(getParamValue('x', node, tensorMap, context))];
        }
        default:
            throw TypeError(`Node type ${node.op} is not implemented`);
    }
};
export const CATEGORY = 'creation';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY3JlYXRpb25fZXhlY3V0b3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvb3BlcmF0aW9ucy9leGVjdXRvcnMvY3JlYXRpb25fZXhlY3V0b3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsaURBQWlEO0FBQ2pELE9BQU8sS0FBSyxLQUFLLE1BQU0sa0RBQWtELENBQUM7QUFNMUUsT0FBTyxFQUFDLGFBQWEsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUV0QyxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQ2xCLENBQUMsSUFBVSxFQUFFLFNBQTBCLEVBQ3RDLE9BQXlCLEVBQVksRUFBRTtJQUN0QyxRQUFRLElBQUksQ0FBQyxFQUFFLEVBQUU7UUFDZixLQUFLLE1BQU0sQ0FBQyxDQUFDO1lBQ1gsTUFBTSxLQUFLLEdBQ1AsYUFBYSxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ2pFLE1BQU0sS0FBSyxHQUNQLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUNqRSxNQUFNLEtBQUssR0FDUCxhQUFhLENBQUMsT0FBTyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDL0QsT0FBTyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDO1NBQzFDO1FBQ0QsS0FBSyxVQUFVLENBQUMsQ0FBQztZQUNmLE1BQU0sS0FBSyxHQUNQLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUMvRCxNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsTUFBTSxHQUFHLEdBQUcsYUFBYSxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ3JFLE9BQU8sQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztTQUMzQztRQUNELEtBQUssYUFBYSxDQUFDLENBQUM7WUFDbEIsTUFBTSxNQUFNLEdBQ1IsYUFBYSxDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBQ2xFLE1BQU0sVUFBVSxHQUNaLGFBQWEsQ0FBQyxZQUFZLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNwRSxNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsT0FBTyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsTUFBTSxFQUFFLFVBQVUsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO1NBQ3REO1FBQ0QsS0FBSyxRQUFRLENBQUMsQ0FBQztZQUNiLE1BQU0sT0FBTyxHQUNULGFBQWEsQ0FBQyxTQUFTLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUNuRSxNQUFNLEtBQUssR0FDUCxhQUFhLENBQUMsT0FBTyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDL0QsTUFBTSxPQUFPLEdBQ1QsYUFBYSxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ2pFLE1BQU0sUUFBUSxHQUNWLGFBQWEsQ0FBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNsRSxPQUFPLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxPQUFPLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBRSxRQUFRLENBQUMsQ0FBQyxDQUFDO1NBQzFEO1FBQ0QsS0FBSyxNQUFNLENBQUMsQ0FBQztZQUNYLE9BQU8sQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUNkLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsRUFDNUQsYUFBYSxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDLENBQUMsQ0FBQztTQUNwRTtRQUNELEtBQUssVUFBVSxDQUFDLENBQUM7WUFDZixPQUFPLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FDbEIsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDLENBQUMsQ0FBQztTQUM5RDtRQUNELEtBQUssZUFBZSxDQUFDLENBQUM7WUFDcEIsT0FBTyxDQUFDLEtBQUssQ0FBQyxhQUFhO2dCQUN2QixrQ0FBa0M7Z0JBQ2xDLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVEsRUFDdkQsYUFBYSxDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxFQUMzRCxhQUFhLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLEVBQzNELGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQyxDQUFDLENBQUM7U0FDcEU7UUFDRCxLQUFLLE9BQU8sQ0FBQyxDQUFDO1lBQ1osTUFBTSxLQUFLLEdBQ1AsYUFBYSxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQy9ELE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUM5RCxNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsT0FBTyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQ2YsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQ2pCLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQ3BDLENBQUMsQ0FBQyxDQUFDO1NBQ25CO1FBQ0QsS0FBSyxpQkFBaUIsQ0FBQyxDQUFDO1lBQ3RCLE1BQU0sS0FBSyxHQUNQLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUNqRSxNQUFNLElBQUksR0FDTixhQUFhLENBQUMsTUFBTSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDOUQsTUFBTSxNQUFNLEdBQ1IsYUFBYSxDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ2hFLE1BQU0sSUFBSSxHQUNOLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUM5RCxPQUFPLENBQUMsS0FBSyxDQUFDLGVBQWUsQ0FDekIsS0FBSyxFQUFFLElBQUksRUFBRSxNQUFNLEVBQ25CLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQ3BDLEVBQ1gsSUFBSSxDQUFDLENBQUMsQ0FBQztTQUNaO1FBQ0QsS0FBSyxPQUFPLENBQUMsQ0FBQztZQUNaLE9BQU8sQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUNmLGFBQWEsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsRUFDNUQsYUFBYSxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDLENBQUMsQ0FBQztTQUNwRTtRQUNELEtBQUssV0FBVyxDQUFDLENBQUM7WUFDaEIsT0FBTyxDQUFDLEtBQUssQ0FBQyxTQUFTLENBQ25CLGFBQWEsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQyxDQUFDLENBQUM7U0FDOUQ7UUFDRDtZQUNFLE1BQU0sU0FBUyxDQUFDLGFBQWEsSUFBSSxDQUFDLEVBQUUscUJBQXFCLENBQUMsQ0FBQztLQUM5RDtBQUNILENBQUMsQ0FBQztBQUVOLE1BQU0sQ0FBQyxNQUFNLFFBQVEsR0FBRyxVQUFVLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RGF0YVR5cGUsIFRlbnNvciwgVGVuc29yMUR9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWltcG9ydHMtZnJvbS1kaXN0XG5pbXBvcnQgKiBhcyB0Zk9wcyBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUvZGlzdC9vcHMvb3BzX2Zvcl9jb252ZXJ0ZXInO1xuXG5pbXBvcnQge05hbWVkVGVuc29yc01hcH0gZnJvbSAnLi4vLi4vZGF0YS90eXBlcyc7XG5pbXBvcnQge0V4ZWN1dGlvbkNvbnRleHR9IGZyb20gJy4uLy4uL2V4ZWN1dG9yL2V4ZWN1dGlvbl9jb250ZXh0JztcbmltcG9ydCB7SW50ZXJuYWxPcEV4ZWN1dG9yLCBOb2RlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7Z2V0UGFyYW1WYWx1ZX0gZnJvbSAnLi91dGlscyc7XG5cbmV4cG9ydCBjb25zdCBleGVjdXRlT3A6IEludGVybmFsT3BFeGVjdXRvciA9XG4gICAgKG5vZGU6IE5vZGUsIHRlbnNvck1hcDogTmFtZWRUZW5zb3JzTWFwLFxuICAgICBjb250ZXh0OiBFeGVjdXRpb25Db250ZXh0KTogVGVuc29yW10gPT4ge1xuICAgICAgc3dpdGNoIChub2RlLm9wKSB7XG4gICAgICAgIGNhc2UgJ0ZpbGwnOiB7XG4gICAgICAgICAgY29uc3Qgc2hhcGUgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzaGFwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG4gICAgICAgICAgY29uc3QgZHR5cGUgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdkdHlwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgRGF0YVR5cGU7XG4gICAgICAgICAgY29uc3QgdmFsdWUgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd2YWx1ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIHJldHVybiBbdGZPcHMuZmlsbChzaGFwZSwgdmFsdWUsIGR0eXBlKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnTGluU3BhY2UnOiB7XG4gICAgICAgICAgY29uc3Qgc3RhcnQgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzdGFydCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IHN0b3AgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzdG9wJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgY29uc3QgbnVtID0gZ2V0UGFyYW1WYWx1ZSgnbnVtJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5saW5zcGFjZShzdGFydCwgc3RvcCwgbnVtKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnTXVsdGlub21pYWwnOiB7XG4gICAgICAgICAgY29uc3QgbG9naXRzID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnbG9naXRzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IxRDtcbiAgICAgICAgICBjb25zdCBudW1TYW1wbGVzID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnbnVtU2FtcGxlcycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IHNlZWQgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzZWVkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5tdWx0aW5vbWlhbChsb2dpdHMsIG51bVNhbXBsZXMsIHNlZWQpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdPbmVIb3QnOiB7XG4gICAgICAgICAgY29uc3QgaW5kaWNlcyA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2luZGljZXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjFEO1xuICAgICAgICAgIGNvbnN0IGRlcHRoID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZGVwdGgnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgICAgICBjb25zdCBvblZhbHVlID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnb25WYWx1ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IG9mZlZhbHVlID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnb2ZmVmFsdWUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgICAgICByZXR1cm4gW3RmT3BzLm9uZUhvdChpbmRpY2VzLCBkZXB0aCwgb25WYWx1ZSwgb2ZmVmFsdWUpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdPbmVzJzoge1xuICAgICAgICAgIHJldHVybiBbdGZPcHMub25lcyhcbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnc2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdLFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdkdHlwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgRGF0YVR5cGUpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdPbmVzTGlrZSc6IHtcbiAgICAgICAgICByZXR1cm4gW3RmT3BzLm9uZXNMaWtlKFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IpXTtcbiAgICAgICAgfVxuICAgICAgICBjYXNlICdSYW5kb21Vbmlmb3JtJzoge1xuICAgICAgICAgIHJldHVybiBbdGZPcHMucmFuZG9tVW5pZm9ybShcbiAgICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzaGFwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYW55LFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdtaW52YWwnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcixcbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnbWF4dmFsJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXIsXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2R0eXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBEYXRhVHlwZSldO1xuICAgICAgICB9XG4gICAgICAgIGNhc2UgJ1JhbmdlJzoge1xuICAgICAgICAgIGNvbnN0IHN0YXJ0ID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnc3RhcnQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgICAgICBjb25zdCBzdG9wID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnc3RvcCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IHN0ZXAgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzdGVwJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy5yYW5nZShcbiAgICAgICAgICAgICAgc3RhcnQsIHN0b3AsIHN0ZXAsXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2R0eXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyAnZmxvYXQzMicgfFxuICAgICAgICAgICAgICAgICAgJ2ludDMyJyldO1xuICAgICAgICB9XG4gICAgICAgIGNhc2UgJ1RydW5jYXRlZE5vcm1hbCc6IHtcbiAgICAgICAgICBjb25zdCBzaGFwZSA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3NoYXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgICAgICBjb25zdCBtZWFuID1cbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnbWVhbicsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IHN0ZERldiA9XG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3N0ZERldicsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgIGNvbnN0IHNlZWQgPVxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdzZWVkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICAgICAgcmV0dXJuIFt0Zk9wcy50cnVuY2F0ZWROb3JtYWwoXG4gICAgICAgICAgICAgIHNoYXBlLCBtZWFuLCBzdGREZXYsXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2R0eXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyAnZmxvYXQzMicgfFxuICAgICAgICAgICAgICAgICAgJ2ludDMyJyxcbiAgICAgICAgICAgICAgc2VlZCldO1xuICAgICAgICB9XG4gICAgICAgIGNhc2UgJ1plcm9zJzoge1xuICAgICAgICAgIHJldHVybiBbdGZPcHMuemVyb3MoXG4gICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3NoYXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXSxcbiAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZHR5cGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIERhdGFUeXBlKV07XG4gICAgICAgIH1cbiAgICAgICAgY2FzZSAnWmVyb3NMaWtlJzoge1xuICAgICAgICAgIHJldHVybiBbdGZPcHMuemVyb3NMaWtlKFxuICAgICAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3IpXTtcbiAgICAgICAgfVxuICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgIHRocm93IFR5cGVFcnJvcihgTm9kZSB0eXBlICR7bm9kZS5vcH0gaXMgbm90IGltcGxlbWVudGVkYCk7XG4gICAgICB9XG4gICAgfTtcblxuZXhwb3J0IGNvbnN0IENBVEVHT1JZID0gJ2NyZWF0aW9uJztcbiJdfQ==