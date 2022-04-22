/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * Padding Layers.
 */
// Porting Note: In Python Keras, the padding layers are in convolutional.py,
//   but we decided to put them in a separate file (padding.ts) for clarity.
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { imageDataFormat } from '../backend/common';
import { InputSpec, Layer } from '../engine/topology';
import { ValueError } from '../errors';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
/**
 * Pads the middle dimension of a 3D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of 2 integers, how many zeros to add at the start and
 *   end of the middle dimension (i.e., dimension 1).
 * @return A padded 3D `tf.Tensor`.
 */
export function temporalPadding(x, padding) {
    return tidy(() => {
        if (x.rank !== 3) {
            throw new ValueError(`temporalPadding expects input tensor to be 3-D, but received a ` +
                `${x.rank}-D tensor.`);
        }
        if (padding == null) {
            padding = [1, 1];
        }
        if (padding.length !== 2) {
            throw new ValueError(`temporalPadding expects input padding pattern to be a length-2 ` +
                `array, but received a length-${padding.length} array.`);
        }
        const pattern = [[0, 0], padding, [0, 0]];
        return tfc.pad(x, pattern);
    });
}
/**
 * Pads the 2nd and 3rd dimensions of a 4D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of two `Array`s, each of which is an `Array` of two
 *   integers. The amount of padding at the beginning and end of the 2nd and 3rd
 *   dimensions, respectively.
 * @param dataFormat 'channelsLast' (default) or 'channelsFirst'.
 * @return Padded 4D `tf.Tensor`.
 */
export function spatial2dPadding(x, padding, dataFormat) {
    return tidy(() => {
        if (x.rank !== 4) {
            throw new ValueError(`temporalPadding expects input tensor to be 4-D, but received a ` +
                `${x.rank}-D tensor.`);
        }
        if (padding == null) {
            padding = [[1, 1], [1, 1]];
        }
        if (padding.length !== 2 || padding[0].length !== 2 ||
            padding[1].length !== 2) {
            throw new ValueError('spatial2dPadding expects `padding` to be an Array of two Arrays, ' +
                'each of which is an Array of two integers.');
        }
        if (dataFormat == null) {
            dataFormat = imageDataFormat();
        }
        if (dataFormat !== 'channelsLast' && dataFormat !== 'channelsFirst') {
            throw new ValueError(`Unknown data format: ${dataFormat}. ` +
                `Supported data formats are 'channelsLast' and 'channelsFirst.`);
        }
        let pattern;
        if (dataFormat === 'channelsFirst') {
            pattern = [[0, 0], [0, 0], padding[0], padding[1]];
        }
        else {
            pattern = [[0, 0], padding[0], padding[1], [0, 0]];
        }
        return tfc.pad(x, pattern);
    });
}
export class ZeroPadding2D extends Layer {
    constructor(args) {
        if (args == null) {
            args = {};
        }
        super(args);
        this.dataFormat =
            args.dataFormat == null ? imageDataFormat() : args.dataFormat;
        // TODO(cais): Maybe refactor the following logic surrounding `padding`
        //   into a helper method.
        if (args.padding == null) {
            this.padding = [[1, 1], [1, 1]];
        }
        else if (typeof args.padding === 'number') {
            this.padding =
                [[args.padding, args.padding], [args.padding, args.padding]];
        }
        else {
            args.padding = args.padding;
            if (args.padding.length !== 2) {
                throw new ValueError(`ZeroPadding2D expects padding to be a length-2 array, but ` +
                    `received a length-${args.padding.length} array.`);
            }
            let heightPadding;
            let widthPadding;
            if (typeof args.padding[0] === 'number') {
                heightPadding = [args.padding[0], args.padding[0]];
                widthPadding = [args.padding[1], args.padding[1]];
            }
            else {
                args.padding = args.padding;
                if (args.padding[0].length !== 2) {
                    throw new ValueError(`ZeroPadding2D expects height padding to be a length-2 array, ` +
                        `but received a length-${args.padding[0].length} array.`);
                }
                heightPadding = args.padding[0];
                if (args.padding[1].length !== 2) {
                    throw new ValueError(`ZeroPadding2D expects width padding to be a length-2 array, ` +
                        `but received a length-${args.padding[1].length} array.`);
                }
                widthPadding = args.padding[1];
            }
            this.padding = [heightPadding, widthPadding];
        }
        this.inputSpec = [new InputSpec({ ndim: 4 })];
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        let rows;
        let cols;
        if (this.dataFormat === 'channelsFirst') {
            if (inputShape[2] != null && inputShape[2] >= 0) {
                rows = inputShape[2] + this.padding[0][0] + this.padding[0][1];
            }
            else {
                rows = null;
            }
            if (inputShape[3] != null && inputShape[3] >= 0) {
                cols = inputShape[3] + this.padding[1][0] + this.padding[1][1];
            }
            else {
                cols = null;
            }
            return [inputShape[0], inputShape[1], rows, cols];
        }
        else {
            if (inputShape[1] != null && inputShape[1] >= 0) {
                rows = inputShape[1] + this.padding[0][0] + this.padding[0][1];
            }
            else {
                rows = null;
            }
            if (inputShape[2] != null && inputShape[2] >= 0) {
                cols = inputShape[2] + this.padding[1][0] + this.padding[1][1];
            }
            else {
                cols = null;
            }
            return [inputShape[0], rows, cols, inputShape[3]];
        }
    }
    call(inputs, kwargs) {
        return tidy(() => spatial2dPadding(getExactlyOneTensor(inputs), this.padding, this.dataFormat));
    }
    getConfig() {
        const config = {
            padding: this.padding,
            dataFormat: this.dataFormat,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
ZeroPadding2D.className = 'ZeroPadding2D';
serialization.registerClass(ZeroPadding2D);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFkZGluZy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvcGFkZGluZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOztHQUVHO0FBRUgsNkVBQTZFO0FBQzdFLDRFQUE0RTtBQUU1RSxPQUFPLEtBQUssR0FBRyxNQUFNLHVCQUF1QixDQUFDO0FBQzdDLE9BQU8sRUFBQyxhQUFhLEVBQVUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFbEUsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ2xELE9BQU8sRUFBQyxTQUFTLEVBQUUsS0FBSyxFQUFZLE1BQU0sb0JBQW9CLENBQUM7QUFDL0QsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUdyQyxPQUFPLEVBQUMsa0JBQWtCLEVBQUUsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUU3RTs7Ozs7OztHQU9HO0FBQ0gsTUFBTSxVQUFVLGVBQWUsQ0FBQyxDQUFTLEVBQUUsT0FBMEI7SUFDbkUsT0FBTyxJQUFJLENBQUMsR0FBRyxFQUFFO1FBQ2YsSUFBSSxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtZQUNoQixNQUFNLElBQUksVUFBVSxDQUNoQixpRUFBaUU7Z0JBQ2pFLEdBQUcsQ0FBQyxDQUFDLElBQUksWUFBWSxDQUFDLENBQUM7U0FDNUI7UUFFRCxJQUFJLE9BQU8sSUFBSSxJQUFJLEVBQUU7WUFDbkIsT0FBTyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1NBQ2xCO1FBQ0QsSUFBSSxPQUFPLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtZQUN4QixNQUFNLElBQUksVUFBVSxDQUNoQixpRUFBaUU7Z0JBQ2pFLGdDQUFnQyxPQUFPLENBQUMsTUFBTSxTQUFTLENBQUMsQ0FBQztTQUM5RDtRQUVELE1B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LENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNuRDtJQUNILENBQUM7SUFFRCxJQUFJLENBQUMsTUFBdUIsRUFBRSxNQUFjO1FBQzFDLE9BQU8sSUFBSSxDQUNQLEdBQUcsRUFBRSxDQUFDLGdCQUFnQixDQUNsQixtQkFBbUIsQ0FBQyxNQUFNLENBQUMsRUFBRSxJQUFJLENBQUMsT0FBTyxFQUFFLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLENBQUM7SUFFRCxTQUFTO1FBQ1AsTUFBTSxNQUFNLEdBQTZCO1lBQ3ZDLE9BQU8sRUFBRSxJQUFJLENBQUMsT0FBTztZQUNyQixVQUFVLEVBQUUsSUFBSSxDQUFDLFVBQVU7U0FDNUIsQ0FBQztRQUNGLE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDOztBQXJHRCxrQkFBa0I7QUFDWCx1QkFBUyxHQUFHLGVBQWUsQ0FBQztBQXNHckMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxhQUFhLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogUGFkZGluZyBMYXllcnMuXG4gKi9cblxuLy8gUG9ydGluZyBOb3RlOiBJbiBQeXRob24gS2VyYXMsIHRoZSBwYWRkaW5nIGxheWVycyBhcmUgaW4gY29udm9sdXRpb25hbC5weSxcbi8vICAgYnV0IHdlIGRlY2lkZWQgdG8gcHV0IHRoZW0gaW4gYSBzZXBhcmF0ZSBmaWxlIChwYWRkaW5nLnRzKSBmb3IgY2xhcml0eS5cblxuaW1wb3J0ICogYXMgdGZjIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge3NlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtpbWFnZURhdGFGb3JtYXR9IGZyb20gJy4uL2JhY2tlbmQvY29tbW9uJztcbmltcG9ydCB7SW5wdXRTcGVjLCBMYXllciwgTGF5ZXJBcmdzfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtEYXRhRm9ybWF0LCBTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge0t3YXJnc30gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtnZXRFeGFjdGx5T25lU2hhcGUsIGdldEV4YWN0bHlPbmVUZW5zb3J9IGZyb20gJy4uL3V0aWxzL3R5cGVzX3V0aWxzJztcblxuLyoqXG4gKiBQYWRzIHRoZSBtaWRkbGUgZGltZW5zaW9uIG9mIGEgM0QgdGVuc29yLlxuICpcbiAqIEBwYXJhbSB4IElucHV0IGB0Zi5UZW5zb3JgIHRvIGJlIHBhZGRlZC5cbiAqIEBwYXJhbSBwYWRkaW5nIGBBcnJheWAgb2YgMiBpbnRlZ2VycywgaG93IG1hbnkgemVyb3MgdG8gYWRkIGF0IHRoZSBzdGFydCBhbmRcbiAqICAgZW5kIG9mIHRoZSBtaWRkbGUgZGltZW5zaW9uIChpLmUuLCBkaW1lbnNpb24gMSkuXG4gKiBAcmV0dXJuIEEgcGFkZGVkIDNEIGB0Zi5UZW5zb3JgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gdGVtcG9yYWxQYWRkaW5nKHg6IFRlbnNvciwgcGFkZGluZz86IFtudW1iZXIsIG51bWJlcl0pOiBUZW5zb3Ige1xuICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgaWYgKHgucmFuayAhPT0gMykge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYHRlbXBvcmFsUGFkZGluZyBleHBlY3RzIGlucHV0IHRlbnNvciB0byBiZSAzLUQsIGJ1dCByZWNlaXZlZCBhIGAgK1xuICAgICAgICAgIGAke3gucmFua30tRCB0ZW5zb3IuYCk7XG4gICAgfVxuXG4gICAgaWYgKHBhZGRpbmcgPT0gbnVsbCkge1xuICAgICAgcGFkZGluZyA9IFsxLCAxXTtcbiAgICB9XG4gICAgaWYgKHBhZGRpbmcubGVuZ3RoICE9PSAyKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgdGVtcG9yYWxQYWRkaW5nIGV4cGVjdHMgaW5wdXQgcGFkZGluZyBwYXR0ZXJuIHRvIGJlIGEgbGVuZ3RoLTIgYCArXG4gICAgICAgICAgYGFycmF5LCBidXQgcmVjZWl2ZWQgYSBsZW5ndGgtJHtwYWRkaW5nLmxlbmd0aH0gYXJyYXkuYCk7XG4gICAgfVxuXG4gICAgY29uc3QgcGF0dGVybjogQXJyYXk8W251bWJlciwgbnVtYmVyXT4gPSBbWzAsIDBdLCBwYWRkaW5nLCBbMCwgMF1dO1xuICAgIHJldHVybiB0ZmMucGFkKHgsIHBhdHRlcm4pO1xuICB9KTtcbn1cblxuLyoqXG4gKiBQYWRzIHRoZSAybmQgYW5kIDNyZCBkaW1lbnNpb25zIG9mIGEgNEQgdGVuc29yLlxuICpcbiAqIEBwYXJhbSB4IElucHV0IGB0Zi5UZW5zb3JgIHRvIGJlIHBhZGRlZC5cbiAqIEBwYXJhbSBwYWRkaW5nIGBBcnJheWAgb2YgdHdvIGBBcnJheWBzLCBlYWNoIG9mIHdoaWNoIGlzIGFuIGBBcnJheWAgb2YgdHdvXG4gKiAgIGludGVnZXJzLiBUaGUgYW1vdW50IG9mIHBhZGRpbmcgYXQgdGhlIGJlZ2lubmluZyBhbmQgZW5kIG9mIHRoZSAybmQgYW5kIDNyZFxuICogICBkaW1lbnNpb25zLCByZXNwZWN0aXZlbHkuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCAnY2hhbm5lbHNMYXN0JyAoZGVmYXVsdCkgb3IgJ2NoYW5uZWxzRmlyc3QnLlxuICogQHJldHVybiBQYWRkZWQgNEQgYHRmLlRlbnNvcmAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzcGF0aWFsMmRQYWRkaW5nKFxuICAgIHg6IFRlbnNvciwgcGFkZGluZz86IFtbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdXSxcbiAgICBkYXRhRm9ybWF0PzogRGF0YUZvcm1hdCk6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICBpZiAoeC5yYW5rICE9PSA0KSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgdGVtcG9yYWxQYWRkaW5nIGV4cGVjdHMgaW5wdXQgdGVuc29yIHRvIGJlIDQtRCwgYnV0IHJlY2VpdmVkIGEgYCArXG4gICAgICAgICAgYCR7eC5yYW5rfS1EIHRlbnNvci5gKTtcbiAgICB9XG5cbiAgICBpZiAocGFkZGluZyA9PSBudWxsKSB7XG4gICAgICBwYWRkaW5nID0gW1sxLCAxXSwgWzEsIDFdXTtcbiAgICB9XG4gICAgaWYgKHBhZGRpbmcubGVuZ3RoICE9PSAyIHx8IHBhZGRpbmdbMF0ubGVuZ3RoICE9PSAyIHx8XG4gICAgICAgIHBhZGRpbmdbMV0ubGVuZ3RoICE9PSAyKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnc3BhdGlhbDJkUGFkZGluZyBleHBlY3RzIGBwYWRkaW5nYCB0byBiZSBhbiBBcnJheSBvZiB0d28gQXJyYXlzLCAnICtcbiAgICAgICAgICAnZWFjaCBvZiB3aGljaCBpcyBhbiBBcnJheSBvZiB0d28gaW50ZWdlcnMuJyk7XG4gICAgfVxuXG4gICAgaWYgKGRhdGFGb3JtYXQgPT0gbnVsbCkge1xuICAgICAgZGF0YUZvcm1hdCA9IGltYWdlRGF0YUZvcm1hdCgpO1xuICAgIH1cbiAgICBpZiAoZGF0YUZvcm1hdCAhPT0gJ2NoYW5uZWxzTGFzdCcgJiYgZGF0YUZvcm1hdCAhPT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICBgVW5rbm93biBkYXRhIGZvcm1hdDogJHtkYXRhRm9ybWF0fS4gYCArXG4gICAgICAgICAgYFN1cHBvcnRlZCBkYXRhIGZvcm1hdHMgYXJlICdjaGFubmVsc0xhc3QnIGFuZCAnY2hhbm5lbHNGaXJzdC5gKTtcbiAgICB9XG5cbiAgICBsZXQgcGF0dGVybjogQXJyYXk8W251bWJlciwgbnVtYmVyXT47XG4gICAgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgICAgcGF0dGVybiA9IFtbMCwgMF0sIFswLCAwXSwgcGFkZGluZ1swXSwgcGFkZGluZ1sxXV07XG4gICAgfSBlbHNlIHtcbiAgICAgIHBhdHRlcm4gPSBbWzAsIDBdLCBwYWRkaW5nWzBdLCBwYWRkaW5nWzFdLCBbMCwgMF1dO1xuICAgIH1cblxuICAgIHJldHVybiB0ZmMucGFkKHgsIHBhdHRlcm4pO1xuICB9KTtcbn1cblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFplcm9QYWRkaW5nMkRMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogSW50ZWdlciwgb3IgYEFycmF5YCBvZiAyIGludGVnZXJzLCBvciBgQXJyYXlgIG9mIDIgYEFycmF5YHMsIGVhY2ggb2ZcbiAgICogd2hpY2ggaXMgYW4gYEFycmF5YCBvZiAyIGludGVnZXJzLlxuICAgKiAtIElmIGludGVnZXIsIHRoZSBzYW1lIHN5bW1ldHJpYyBwYWRkaW5nIGlzIGFwcGxpZWQgdG8gd2lkdGggYW5kIGhlaWdodC5cbiAgICogLSBJZiBBcnJheWAgb2YgMiBpbnRlZ2VycywgaW50ZXJwcmV0ZWQgYXMgdHdvIGRpZmZlcmVudCBzeW1tZXRyaWMgdmFsdWVzXG4gICAqICAgZm9yIGhlaWdodCBhbmQgd2lkdGg6XG4gICAqICAgYFtzeW1tZXRyaWNIZWlnaHRQYWQsIHN5bW1ldHJpY1dpZHRoUGFkXWAuXG4gICAqIC0gSWYgYEFycmF5YCBvZiAyIGBBcnJheWBzLCBpbnRlcnByZXRlZCBhczpcbiAgICogICBgW1t0b3BQYWQsIGJvdHRvbVBhZF0sIFtsZWZ0UGFkLCByaWdodFBhZF1dYC5cbiAgICovXG4gIHBhZGRpbmc/OiBudW1iZXJ8W251bWJlciwgbnVtYmVyXXxbW251bWJlciwgbnVtYmVyXSwgW251bWJlciwgbnVtYmVyXV07XG5cbiAgLyoqXG4gICAqIE9uZSBvZiBgJ2NoYW5uZWxzTGFzdCdgIChkZWZhdWx0KSBhbmQgYCdjaGFubmVsc0ZpcnN0J2AuXG4gICAqXG4gICAqIFRoZSBvcmRlcmluZyBvZiB0aGUgZGltZW5zaW9ucyBpbiB0aGUgaW5wdXRzLlxuICAgKiBgY2hhbm5lbHNMYXN0YCBjb3JyZXNwb25kcyB0byBpbnB1dHMgd2l0aCBzaGFwZVxuICAgKiBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc11gIHdoaWxlIGBjaGFubmVsc0ZpcnN0YFxuICAgKiBjb3JyZXNwb25kcyB0byBpbnB1dHMgd2l0aCBzaGFwZVxuICAgKiBgW2JhdGNoLCBjaGFubmVscywgaGVpZ2h0LCB3aWR0aF1gLlxuICAgKi9cbiAgZGF0YUZvcm1hdD86IERhdGFGb3JtYXQ7XG59XG5cbmV4cG9ydCBjbGFzcyBaZXJvUGFkZGluZzJEIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdaZXJvUGFkZGluZzJEJztcbiAgcmVhZG9ubHkgZGF0YUZvcm1hdDogRGF0YUZvcm1hdDtcbiAgcmVhZG9ubHkgcGFkZGluZzogW1tudW1iZXIsIG51bWJlcl0sIFtudW1iZXIsIG51bWJlcl1dO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBaZXJvUGFkZGluZzJETGF5ZXJBcmdzKSB7XG4gICAgaWYgKGFyZ3MgPT0gbnVsbCkge1xuICAgICAgYXJncyA9IHt9O1xuICAgIH1cbiAgICBzdXBlcihhcmdzKTtcblxuICAgIHRoaXMuZGF0YUZvcm1hdCA9XG4gICAgICAgIGFyZ3MuZGF0YUZvcm1hdCA9PSBudWxsID8gaW1hZ2VEYXRhRm9ybWF0KCkgOiBhcmdzLmRhdGFGb3JtYXQ7XG4gICAgLy8gVE9ETyhjYWlzKTogTWF5YmUgcmVmYWN0b3IgdGhlIGZvbGxvd2luZyBsb2dpYyBzdXJyb3VuZGluZyBgcGFkZGluZ2BcbiAgICAvLyAgIGludG8gYSBoZWxwZXIgbWV0aG9kLlxuICAgIGlmIChhcmdzLnBhZGRpbmcgPT0gbnVsbCkge1xuICAgICAgdGhpcy5wYWRkaW5nID0gW1sxLCAxXSwgWzEsIDFdXTtcbiAgICB9IGVsc2UgaWYgKHR5cGVvZiBhcmdzLnBhZGRpbmcgPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLnBhZGRpbmcgPVxuICAgICAgICAgIFtbYXJncy5wYWRkaW5nLCBhcmdzLnBhZGRpbmddLCBbYXJncy5wYWRkaW5nLCBhcmdzLnBhZGRpbmddXTtcbiAgICB9IGVsc2Uge1xuICAgICAgYXJncy5wYWRkaW5nID0gYXJncy5wYWRkaW5nO1xuICAgICAgaWYgKGFyZ3MucGFkZGluZy5sZW5ndGggIT09IDIpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgWmVyb1BhZGRpbmcyRCBleHBlY3RzIHBhZGRpbmcgdG8gYmUgYSBsZW5ndGgtMiBhcnJheSwgYnV0IGAgK1xuICAgICAgICAgICAgYHJlY2VpdmVkIGEgbGVuZ3RoLSR7YXJncy5wYWRkaW5nLmxlbmd0aH0gYXJyYXkuYCk7XG4gICAgICB9XG5cbiAgICAgIGxldCBoZWlnaHRQYWRkaW5nOiBbbnVtYmVyLCBudW1iZXJdO1xuICAgICAgbGV0IHdpZHRoUGFkZGluZzogW251bWJlciwgbnVtYmVyXTtcbiAgICAgIGlmICh0eXBlb2YgYXJncy5wYWRkaW5nWzBdID09PSAnbnVtYmVyJykge1xuICAgICAgICBoZWlnaHRQYWRkaW5nID0gW2FyZ3MucGFkZGluZ1swXSwgYXJncy5wYWRkaW5nWzBdXTtcbiAgICAgICAgd2lkdGhQYWRkaW5nID0gW2FyZ3MucGFkZGluZ1sxXSBhcyBudW1iZXIsIGFyZ3MucGFkZGluZ1sxXSBhcyBudW1iZXJdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYXJncy5wYWRkaW5nID0gYXJncy5wYWRkaW5nIGFzIFtbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdXTtcblxuICAgICAgICBpZiAoYXJncy5wYWRkaW5nWzBdLmxlbmd0aCAhPT0gMikge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICBgWmVyb1BhZGRpbmcyRCBleHBlY3RzIGhlaWdodCBwYWRkaW5nIHRvIGJlIGEgbGVuZ3RoLTIgYXJyYXksIGAgK1xuICAgICAgICAgICAgICBgYnV0IHJlY2VpdmVkIGEgbGVuZ3RoLSR7YXJncy5wYWRkaW5nWzBdLmxlbmd0aH0gYXJyYXkuYCk7XG4gICAgICAgIH1cbiAgICAgICAgaGVpZ2h0UGFkZGluZyA9IGFyZ3MucGFkZGluZ1swXSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuXG4gICAgICAgIGlmIChhcmdzLnBhZGRpbmdbMV0ubGVuZ3RoICE9PSAyKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBaZXJvUGFkZGluZzJEIGV4cGVjdHMgd2lkdGggcGFkZGluZyB0byBiZSBhIGxlbmd0aC0yIGFycmF5LCBgICtcbiAgICAgICAgICAgICAgYGJ1dCByZWNlaXZlZCBhIGxlbmd0aC0ke2FyZ3MucGFkZGluZ1sxXS5sZW5ndGh9IGFycmF5LmApO1xuICAgICAgICB9XG4gICAgICAgIHdpZHRoUGFkZGluZyA9IGFyZ3MucGFkZGluZ1sxXSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuICAgICAgfVxuICAgICAgdGhpcy5wYWRkaW5nID0gW2hlaWdodFBhZGRpbmcsIHdpZHRoUGFkZGluZ107XG4gICAgfVxuICAgIHRoaXMuaW5wdXRTcGVjID0gW25ldyBJbnB1dFNwZWMoe25kaW06IDR9KV07XG4gIH1cblxuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG5cbiAgICBsZXQgcm93czogbnVtYmVyO1xuICAgIGxldCBjb2xzOiBudW1iZXI7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgICBpZiAoaW5wdXRTaGFwZVsyXSAhPSBudWxsICYmIGlucHV0U2hhcGVbMl0gPj0gMCkge1xuICAgICAgICByb3dzID0gaW5wdXRTaGFwZVsyXSArIHRoaXMucGFkZGluZ1swXVswXSArIHRoaXMucGFkZGluZ1swXVsxXTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJvd3MgPSBudWxsO1xuICAgICAgfVxuICAgICAgaWYgKGlucHV0U2hhcGVbM10gIT0gbnVsbCAmJiBpbnB1dFNoYXBlWzNdID49IDApIHtcbiAgICAgICAgY29scyA9IGlucHV0U2hhcGVbM10gKyB0aGlzLnBhZGRpbmdbMV1bMF0gKyB0aGlzLnBhZGRpbmdbMV1bMV07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb2xzID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIHJldHVybiBbaW5wdXRTaGFwZVswXSwgaW5wdXRTaGFwZVsxXSwgcm93cywgY29sc107XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmIChpbnB1dFNoYXBlWzFdICE9IG51bGwgJiYgaW5wdXRTaGFwZVsxXSA+PSAwKSB7XG4gICAgICAgIHJvd3MgPSBpbnB1dFNoYXBlWzFdICsgdGhpcy5wYWRkaW5nWzBdWzBdICsgdGhpcy5wYWRkaW5nWzBdWzFdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcm93cyA9IG51bGw7XG4gICAgICB9XG4gICAgICBpZiAoaW5wdXRTaGFwZVsyXSAhPSBudWxsICYmIGlucHV0U2hhcGVbMl0gPj0gMCkge1xuICAgICAgICBjb2xzID0gaW5wdXRTaGFwZVsyXSArIHRoaXMucGFkZGluZ1sxXVswXSArIHRoaXMucGFkZGluZ1sxXVsxXTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbHMgPSBudWxsO1xuICAgICAgfVxuICAgICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCByb3dzLCBjb2xzLCBpbnB1dFNoYXBlWzNdXTtcbiAgICB9XG4gIH1cblxuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRpZHkoXG4gICAgICAgICgpID0+IHNwYXRpYWwyZFBhZGRpbmcoXG4gICAgICAgICAgICBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyksIHRoaXMucGFkZGluZywgdGhpcy5kYXRhRm9ybWF0KSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHtcbiAgICAgIHBhZGRpbmc6IHRoaXMucGFkZGluZyxcbiAgICAgIGRhdGFGb3JtYXQ6IHRoaXMuZGF0YUZvcm1hdCxcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFplcm9QYWRkaW5nMkQpO1xuIl19