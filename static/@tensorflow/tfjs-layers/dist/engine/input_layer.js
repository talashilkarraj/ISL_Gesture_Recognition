/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { serialization } from '@tensorflow/tfjs-core';
import { getUid } from '../backend/state';
import { ValueError } from '../errors';
import { Layer, Node, SymbolicTensor } from './topology';
export class InputLayer extends Layer {
    constructor(args) {
        super({
            dtype: args.dtype,
            name: args.name != null ? args.name : getUid('input').toString()
        });
        // Normalize config.batchSize and config.sparse
        if (args.batchSize == null) {
            args.batchSize = null;
        }
        if (args.sparse == null) {
            args.sparse = false;
        }
        this.trainable = false;
        this.built = true;
        this.sparse = args.sparse;
        if (args.inputShape != null && args.batchInputShape != null) {
            throw new ValueError('Only provide the inputShape OR ' +
                'batchInputShape argument to inputLayer, not both at the same time.');
        }
        let batchInputShape = args.batchInputShape;
        if (batchInputShape == null) {
            if (args.inputShape == null) {
                throw new ValueError('An InputLayer should be passed either a ' +
                    '`batchInputShape` or an `inputShape`.');
            }
            else {
                batchInputShape = [args.batchSize].concat(args.inputShape);
            }
        }
        else {
            // TODO(michaelterry): Backport to PyKeras
            if (args.batchSize != null) {
                throw new ValueError('Cannot specify batchSize if batchInputShape is ' +
                    'specified when creating an InputLayer.');
            }
        }
        const dtype = args.dtype || 'float32';
        this.batchInputShape = batchInputShape;
        this.dtype = dtype;
        // TODO(michaelterry): Backport this to PyKeras?
        this.inputSpec = [{ shape: batchInputShape }];
        const inputTensor = new SymbolicTensor(this.dtype, this.batchInputShape, this, [], {}, this.name);
        inputTensor.nodeIndex = 0;
        inputTensor.tensorIndex = 0;
        // Create an input node to add to this.outboundNode.
        // (This call has side effects.)
        // tslint:disable-next-line:no-unused-expression
        new Node({
            outboundLayer: this,
            inboundLayers: [],
            nodeIndices: [],
            tensorIndices: [],
            inputTensors: [inputTensor],
            outputTensors: [inputTensor],
            inputMasks: [null],
            outputMasks: [null],
            inputShapes: [batchInputShape],
            outputShapes: [batchInputShape]
        });
    }
    apply(inputs, kwargs) {
        throw new ValueError('Cannot pass any input to an ' +
            `InputLayer's apply() method. InputLayer name: ${this.name}`);
    }
    dispose() {
        // dispose() for InputLayer is overridden as no-op.
        return { refCountAfterDispose: this._refCount, numDisposedVariables: 0 };
    }
    getConfig() {
        return {
            batchInputShape: this.batchInputShape,
            dtype: this.dtype,
            sparse: this.sparse,
            name: this.name
        };
    }
}
/** @nocollapse */
InputLayer.className = 'InputLayer';
serialization.registerClass(InputLayer);
export function Input(config) {
    if (config.batchShape == null && config.shape == null) {
        throw new Error('Please provide to Input either a `shape`' +
            ' or a `batchShape` argument. Note that ' +
            '`shape` does not include the batch ' +
            'dimension.');
    }
    if (config.batchShape != null && config.shape != null) {
        // TODO(michaelterry): Backport to PyKeras.
        throw new ValueError('Please provide either a `shape` or `batchShape` ' +
            'argument to Input, but not both.');
    }
    let batchShape = config.batchShape;
    if (config.shape != null && batchShape == null) {
        batchShape = [null].concat(config.shape);
    }
    let dtype = config.dtype;
    if (dtype == null) {
        dtype = 'float32';
    }
    const inputLayer = new InputLayer({
        batchInputShape: batchShape,
        name: config.name,
        dtype,
        sparse: config.sparse
    });
    const outputs = inputLayer.inboundNodes[0].outputTensors;
    return outputs[0];
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5wdXRfbGF5ZXIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZW5naW5lL2lucHV0X2xheWVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUgsT0FBTyxFQUFXLGFBQWEsRUFBUyxNQUFNLHVCQUF1QixDQUFDO0FBRXRFLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUN4QyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBSXJDLE9BQU8sRUFBZ0IsS0FBSyxFQUFFLElBQUksRUFBRSxjQUFjLEVBQUMsTUFBTSxZQUFZLENBQUM7QUEyQnRFLE1BQU0sT0FBTyxVQUFXLFNBQVEsS0FBSztJQUluQyxZQUFZLElBQW9CO1FBQzlCLEtBQUssQ0FBQztZQUNKLEtBQUssRUFBRSxJQUFJLENBQUMsS0FBSztZQUNqQixJQUFJLEVBQUUsSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQyxRQUFRLEVBQUU7U0FDakUsQ0FBQyxDQUFDO1FBQ0gsK0NBQStDO1FBQy9DLElBQUksSUFBSSxDQUFDLFNBQVMsSUFBSSxJQUFJLEVBQUU7WUFDMUIsSUFBSSxDQUFDLFNBQVMsR0FBRyxJQUFJLENBQUM7U0FDdkI7UUFDRCxJQUFJLElBQUksQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFFO1lBQ3ZCLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1NBQ3JCO1FBRUQsSUFBSSxDQUFDLFNBQVMsR0FBRyxLQUFLLENBQUM7UUFDdkIsSUFBSSxDQUFDLEtBQUssR0FBRyxJQUFJLENBQUM7UUFDbEIsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDO1FBRTFCLElBQUksSUFBSSxDQUFDLFVB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QyxVQUFVLElBQUksSUFBSSxJQUFJLE1BQU0sQ0FBQyxLQUFLLElBQUksSUFBSSxFQUFFO1FBQ3JELE1BQU0sSUFBSSxLQUFLLENBQ1gsMENBQTBDO1lBQzFDLHlDQUF5QztZQUN6QyxxQ0FBcUM7WUFDckMsWUFBWSxDQUFDLENBQUM7S0FDbkI7SUFDRCxJQUFJLE1BQU0sQ0FBQyxVQUFVLElBQUksSUFBSSxJQUFJLE1BQU0sQ0FBQyxLQUFLLElBQUksSUFBSSxFQUFFO1FBQ3JELDJDQUEyQztRQUMzQyxNQUFNLElBQUksVUFBVSxDQUNoQixrREFBa0Q7WUFDbEQsa0NBQWtDLENBQUMsQ0FBQztLQUN6QztJQUNELElBQUksVUFBVSxHQUFHLE1BQU0sQ0FBQyxVQUFVLENBQUM7SUFDbkMsSUFBSSxNQUFNLENBQUMsS0FBSyxJQUFJLElBQUksSUFBSSxVQUFVLElBQUksSUFBSSxFQUFFO1FBQzlDLFVBQVUsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7S0FDMUM7SUFFRCxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDO0lBQ3pCLElBQUksS0FBSyxJQUFJLElBQUksRUFBRTtRQUNqQixLQUFLLEdBQUcsU0FBUyxDQUFDO0tBQ25CO0lBRUQsTUFBTSxVQUFVLEdBQUcsSUFBSSxVQUFVLENBQUM7UUFDaEMsZUFBZSxFQUFFLFVBQVU7UUFDM0IsSUFBSSxFQUFFLE1BQU0sQ0FBQyxJQUFJO1FBQ2pCLEtBQUs7UUFDTCxNQUFNLEVBQUUsTUFBTSxDQUFDLE1BQU07S0FDdEIsQ0FBQyxDQUFDO0lBRUgsTUFBTSxPQUFPLEdBQUcsVUFBVSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQyxhQUFhLENBQUM7SUFDekQsT0FBTyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDcEIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RGF0YVR5cGUsIHNlcmlhbGl6YXRpb24sIFRlbnNvcn0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtnZXRVaWR9IGZyb20gJy4uL2JhY2tlbmQvc3RhdGUnO1xuaW1wb3J0IHtWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge0t3YXJnc30gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge0Rpc3Bvc2VSZXN1bHQsIExheWVyLCBOb2RlLCBTeW1ib2xpY1RlbnNvcn0gZnJvbSAnLi90b3BvbG9neSc7XG5cbi8qKlxuICogQ29uc3RydWN0b3IgYXJndW1lbnRzIGZvciBJbnB1dExheWVyLlxuICpcbiAqIE5vdGU6IFlvdSBzaG91bGQgcHJvdmlkZSBvbmx5IGlucHV0U2hhcGUgb3IgYmF0Y2hJbnB1dFNoYXBlIChub3QgYm90aCkuXG4gKiBJZiBvbmx5IGlucHV0U2hhcGUgaXMgcHJvdmlkZWQsIHRoZW4gdGhlIGJhdGNoSW5wdXRTaGFwZSBpcyBkZXRlcm1pbmVkIGJ5XG4gKiB0aGUgYmF0Y2hTaXplIGFyZ3VtZW50IGFuZCB0aGUgaW5wdXRTaGFwZTogW2JhdGNoU2l6ZV0uY29uY2F0KGlucHV0U2hhcGUpLlxuICovXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgSW5wdXRMYXllckFyZ3Mge1xuICAvKiogSW5wdXQgc2hhcGUsIG5vdCBpbmNsdWRpbmcgdGhlIGJhdGNoIGF4aXMuICovXG4gIGlucHV0U2hhcGU/OiBTaGFwZTtcbiAgLyoqIE9wdGlvbmFsIGlucHV0IGJhdGNoIHNpemUgKGludGVnZXIgb3IgbnVsbCkuICovXG4gIGJhdGNoU2l6ZT86IG51bWJlcjtcbiAgLyoqIEJhdGNoIGlucHV0IHNoYXBlLCBpbmNsdWRpbmcgdGhlIGJhdGNoIGF4aXMuICovXG4gIGJhdGNoSW5wdXRTaGFwZT86IFNoYXBlO1xuICAvKiogRGF0YXR5cGUgb2YgdGhlIGlucHV0LiAgKi9cbiAgZHR5cGU/OiBEYXRhVHlwZTtcbiAgLyoqXG4gICAqIFdoZXRoZXIgdGhlIHBsYWNlaG9sZGVyIGNyZWF0ZWQgaXMgbWVhbnQgdG8gYmUgc3BhcnNlLlxuICAgKi9cbiAgc3BhcnNlPzogYm9vbGVhbjsgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogTm90IGNsZWFyIHdoZXRoZXIgd2UnbGwgbmVlZCB0aGlzLlxuXG4gIC8qKiBOYW1lIG9mIHRoZSBsYXllci4gKi9cbiAgbmFtZT86IHN0cmluZztcbn1cblxuZXhwb3J0IGNsYXNzIElucHV0TGF5ZXIgZXh0ZW5kcyBMYXllciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgcmVhZG9ubHkgY2xhc3NOYW1lID0gJ0lucHV0TGF5ZXInO1xuICBzcGFyc2U6IGJvb2xlYW47XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IElucHV0TGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoe1xuICAgICAgZHR5cGU6IGFyZ3MuZHR5cGUsXG4gICAgICBuYW1lOiBhcmdzLm5hbWUgIT0gbnVsbCA/IGFyZ3MubmFtZSA6IGdldFVpZCgnaW5wdXQnKS50b1N0cmluZygpXG4gICAgfSk7XG4gICAgLy8gTm9ybWFsaXplIGNvbmZpZy5iYXRjaFNpemUgYW5kIGNvbmZpZy5zcGFyc2VcbiAgICBpZiAoYXJncy5iYXRjaFNpemUgPT0gbnVsbCkge1xuICAgICAgYXJncy5iYXRjaFNpemUgPSBudWxsO1xuICAgIH1cbiAgICBpZiAoYXJncy5zcGFyc2UgPT0gbnVsbCkge1xuICAgICAgYXJncy5zcGFyc2UgPSBmYWxzZTtcbiAgICB9XG5cbiAgICB0aGlzLnRyYWluYWJsZSA9IGZhbHNlO1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICAgIHRoaXMuc3BhcnNlID0gYXJncy5zcGFyc2U7XG5cbiAgICBpZiAoYXJncy5pbnB1dFNoYXBlICE9IG51bGwgJiYgYXJncy5iYXRjaElucHV0U2hhcGUgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgJ09ubHkgcHJvdmlkZSB0aGUgaW5wdXRTaGFwZSBPUiAnICtcbiAgICAgICAgICAnYmF0Y2hJbnB1dFNoYXBlIGFyZ3VtZW50IHRvIGlucHV0TGF5ZXIsIG5vdCBib3RoIGF0IHRoZSBzYW1lIHRpbWUuJyk7XG4gICAgfVxuICAgIGxldCBiYXRjaElucHV0U2hhcGUgPSBhcmdzLmJhdGNoSW5wdXRTaGFwZTtcbiAgICBpZiAoYmF0Y2hJbnB1dFNoYXBlID09IG51bGwpIHtcbiAgICAgIGlmIChhcmdzLmlucHV0U2hhcGUgPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICdBbiBJbnB1dExheWVyIHNob3VsZCBiZSBwYXNzZWQgZWl0aGVyIGEgJyArXG4gICAgICAgICAgICAnYGJhdGNoSW5wdXRTaGFwZWAgb3IgYW4gYGlucHV0U2hhcGVgLicpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYmF0Y2hJbnB1dFNoYXBlID0gW2FyZ3MuYmF0Y2hTaXplXS5jb25jYXQoYXJncy5pbnB1dFNoYXBlKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBCYWNrcG9ydCB0byBQeUtlcmFzXG4gICAgICBpZiAoYXJncy5iYXRjaFNpemUgIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICdDYW5ub3Qgc3BlY2lmeSBiYXRjaFNpemUgaWYgYmF0Y2hJbnB1dFNoYXBlIGlzICcgK1xuICAgICAgICAgICAgJ3NwZWNpZmllZCB3aGVuIGNyZWF0aW5nIGFuIElucHV0TGF5ZXIuJyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgZHR5cGUgPSBhcmdzLmR0eXBlIHx8ICdmbG9hdDMyJztcblxuICAgIHRoaXMuYmF0Y2hJbnB1dFNoYXBlID0gYmF0Y2hJbnB1dFNoYXBlO1xuICAgIHRoaXMuZHR5cGUgPSBkdHlwZTtcbiAgICAvLyBUT0RPKG1pY2hhZWx0ZXJyeSk6IEJhY2twb3J0IHRoaXMgdG8gUHlLZXJhcz9cbiAgICB0aGlzLmlucHV0U3BlYyA9IFt7c2hhcGU6IGJhdGNoSW5wdXRTaGFwZX1dO1xuXG4gICAgY29uc3QgaW5wdXRUZW5zb3IgPSBuZXcgU3ltYm9saWNUZW5zb3IoXG4gICAgICAgIHRoaXMuZHR5cGUsIHRoaXMuYmF0Y2hJbnB1dFNoYXBlLCB0aGlzLCBbXSwge30sIHRoaXMubmFtZSk7XG4gICAgaW5wdXRUZW5zb3Iubm9kZUluZGV4ID0gMDtcbiAgICBpbnB1dFRlbnNvci50ZW5zb3JJbmRleCA9IDA7XG5cbiAgICAvLyBDcmVhdGUgYW4gaW5wdXQgbm9kZSB0byBhZGQgdG8gdGhpcy5vdXRib3VuZE5vZGUuXG4gICAgLy8gKFRoaXMgY2FsbCBoYXMgc2lkZSBlZmZlY3RzLilcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tdW51c2VkLWV4cHJlc3Npb25cbiAgICBuZXcgTm9kZSh7XG4gICAgICBvdXRib3VuZExheWVyOiB0aGlzLFxuICAgICAgaW5ib3VuZExheWVyczogW10sXG4gICAgICBub2RlSW5kaWNlczogW10sXG4gICAgICB0ZW5zb3JJbmRpY2VzOiBbXSxcbiAgICAgIGlucHV0VGVuc29yczogW2lucHV0VGVuc29yXSxcbiAgICAgIG91dHB1dFRlbnNvcnM6IFtpbnB1dFRlbnNvcl0sXG4gICAgICBpbnB1dE1hc2tzOiBbbnVsbF0sXG4gICAgICBvdXRwdXRNYXNrczogW251bGxdLFxuICAgICAgaW5wdXRTaGFwZXM6IFtiYXRjaElucHV0U2hhcGVdLFxuICAgICAgb3V0cHV0U2hhcGVzOiBbYmF0Y2hJbnB1dFNoYXBlXVxuICAgIH0pO1xuICB9XG5cbiAgYXBwbHkoXG4gICAgICBpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdLFxuICAgICAga3dhcmdzPzogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yIHtcbiAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgJ0Nhbm5vdCBwYXNzIGFueSBpbnB1dCB0byBhbiAnICtcbiAgICAgICAgYElucHV0TGF5ZXIncyBhcHBseSgpIG1ldGhvZC4gSW5wdXRMYXllciBuYW1lOiAke3RoaXMubmFtZX1gKTtcbiAgfVxuXG4gIGRpc3Bvc2UoKTogRGlzcG9zZVJlc3VsdCB7XG4gICAgLy8gZGlzcG9zZSgpIGZvciBJbnB1dExheWVyIGlzIG92ZXJyaWRkZW4gYXMgbm8tb3AuXG4gICAgcmV0dXJuIHtyZWZDb3VudEFmdGVyRGlzcG9zZTogdGhpcy5fcmVmQ291bnQsIG51bURpc3Bvc2VkVmFyaWFibGVzOiAwfTtcbiAgfVxuXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7XG4gICAgICBiYXRjaElucHV0U2hhcGU6IHRoaXMuYmF0Y2hJbnB1dFNoYXBlLFxuICAgICAgZHR5cGU6IHRoaXMuZHR5cGUsXG4gICAgICBzcGFyc2U6IHRoaXMuc3BhcnNlLFxuICAgICAgbmFtZTogdGhpcy5uYW1lXG4gICAgfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKElucHV0TGF5ZXIpO1xuXG4vKipcbiAqIENvbmZpZyBmb3IgdGhlIElucHV0IGZ1bmN0aW9uLlxuICpcbiAqIE5vdGU6IFlvdSBzaG91bGQgcHJvdmlkZSBvbmx5IHNoYXBlIG9yIGJhdGNoU2hhcGUgKG5vdCBib3RoKS5cbiAqIElmIG9ubHkgc2hhcGUgaXMgcHJvdmlkZWQsIHRoZW4gdGhlIGJhdGNoU2hhcGUgYmVjb21lc1xuICogW251bGxdLmNvbmNhdChpbnB1dFNoYXBlKS5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBJbnB1dENvbmZpZyB7XG4gIC8qKlxuICAgKiBBIHNoYXBlLCBub3QgaW5jbHVkaW5nIHRoZSBiYXRjaCBzaXplLiBGb3IgaW5zdGFuY2UsIGBzaGFwZT1bMzJdYFxuICAgKiBpbmRpY2F0ZXMgdGhhdCB0aGUgZXhwZWN0ZWQgaW5wdXQgd2lsbCBiZSBiYXRjaGVzIG9mIDMyLWRpbWVuc2lvbmFsXG4gICAqIHZlY3RvcnMuXG4gICAqL1xuICBzaGFwZT86IFNoYXBlO1xuICAvKipcbiAgICogQSBzaGFwZSB0dXBsZSAoaW50ZWdlciksIGluY2x1ZGluZyB0aGUgYmF0Y2ggc2l6ZS4gRm9yIGluc3RhbmNlLFxuICAgKiBgYmF0Y2hTaGFwZT1bMTAsIDMyXWAgaW5kaWNhdGVzIHRoYXQgdGhlIGV4cGVjdGVkIGlucHV0IHdpbGwgYmUgYmF0Y2hlcyBvZlxuICAgKiAxMCAzMi1kaW1lbnNpb25hbCB2ZWN0b3JzLiBgYmF0Y2hTaGFwZT1bbnVsbCwgMzJdYCBpbmRpY2F0ZXMgYmF0Y2hlcyBvZiBhblxuICAgKiBhcmJpdHJhcnkgbnVtYmVyIG9mIDMyLWRpbWVuc2lvbmFsIHZlY3RvcnMuXG4gICAqL1xuICBiYXRjaFNoYXBlPzogU2hhcGU7XG4gIC8qKlxuICAgKiBBbiBvcHRpb25hbCBuYW1lIHN0cmluZyBmb3IgdGhlIGxheWVyLiBTaG91bGQgYmUgdW5pcXVlIGluIGEgbW9kZWwgKGRvIG5vdFxuICAgKiByZXVzZSB0aGUgc2FtZSBuYW1lIHR3aWNlKS4gSXQgd2lsbCBiZSBhdXRvZ2VuZXJhdGVkIGlmIGl0IGlzbid0IHByb3ZpZGVkLlxuICAgKi9cbiAgbmFtZT86IHN0cmluZztcbiAgZHR5cGU/OiBEYXRhVHlwZTtcbiAgLyoqXG4gICAqIEEgYm9vbGVhbiBzcGVjaWZ5aW5nIHdoZXRoZXIgdGhlIHBsYWNlaG9sZGVyIHRvIGJlIGNyZWF0ZWQgaXMgc3BhcnNlLlxuICAgKi9cbiAgc3BhcnNlPzogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIElucHV0KGNvbmZpZzogSW5wdXRDb25maWcpOiBTeW1ib2xpY1RlbnNvciB7XG4gIGlmIChjb25maWcuYmF0Y2hTaGFwZSA9PSBudWxsICYmIGNvbmZpZy5zaGFwZSA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnUGxlYXNlIHByb3ZpZGUgdG8gSW5wdXQgZWl0aGVyIGEgYHNoYXBlYCcgK1xuICAgICAgICAnIG9yIGEgYGJhdGNoU2hhcGVgIGFyZ3VtZW50LiBOb3RlIHRoYXQgJyArXG4gICAgICAgICdgc2hhcGVgIGRvZXMgbm90IGluY2x1ZGUgdGhlIGJhdGNoICcgK1xuICAgICAgICAnZGltZW5zaW9uLicpO1xuICB9XG4gIGlmIChjb25maWcuYmF0Y2hTaGFwZSAhPSBudWxsICYmIGNvbmZpZy5zaGFwZSAhPSBudWxsKSB7XG4gICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBCYWNrcG9ydCB0byBQeUtlcmFzLlxuICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAnUGxlYXNlIHByb3ZpZGUgZWl0aGVyIGEgYHNoYXBlYCBvciBgYmF0Y2hTaGFwZWAgJyArXG4gICAgICAgICdhcmd1bWVudCB0byBJbnB1dCwgYnV0IG5vdCBib3RoLicpO1xuICB9XG4gIGxldCBiYXRjaFNoYXBlID0gY29uZmlnLmJhdGNoU2hhcGU7XG4gIGlmIChjb25maWcuc2hhcGUgIT0gbnVsbCAmJiBiYXRjaFNoYXBlID09IG51bGwpIHtcbiAgICBiYXRjaFNoYXBlID0gW251bGxdLmNvbmNhdChjb25maWcuc2hhcGUpO1xuICB9XG5cbiAgbGV0IGR0eXBlID0gY29uZmlnLmR0eXBlO1xuICBpZiAoZHR5cGUgPT0gbnVsbCkge1xuICAgIGR0eXBlID0gJ2Zsb2F0MzInO1xuICB9XG5cbiAgY29uc3QgaW5wdXRMYXllciA9IG5ldyBJbnB1dExheWVyKHtcbiAgICBiYXRjaElucHV0U2hhcGU6IGJhdGNoU2hhcGUsXG4gICAgbmFtZTogY29uZmlnLm5hbWUsXG4gICAgZHR5cGUsXG4gICAgc3BhcnNlOiBjb25maWcuc3BhcnNlXG4gIH0pO1xuXG4gIGNvbnN0IG91dHB1dHMgPSBpbnB1dExheWVyLmluYm91bmROb2Rlc1swXS5vdXRwdXRUZW5zb3JzO1xuICByZXR1cm4gb3V0cHV0c1swXTtcbn1cbiJdfQ==