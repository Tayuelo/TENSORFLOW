import * as tf from "@tensorflow/tfjs";

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
const TRUE_BOUNDING_BOX_STYLE = "rgb(255,0,0)";
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
const PREDICT_BOUNDING_BOX_STYLE = "rgb(0,0,255)";

const generateRandomColorStyle = () =>
  `rgb(${Math.round(Math.random() * 255)},
  ${Math.round(Math.random() * 255)},
  ${Math.round(Math.random() * 255)})`;

const getRandomInt = (min, max) =>
  Math.floor(Math.random() * (max - min)) + min;

class ObjectDetectionImageSynthesizer {
  constructor(canvas) {
    this.canvas = canvas;

    // Min and max of circles' radii.
    this.CIRCLE_RADIUS_MIN = 5;
    this.CIRCLE_RADIUS_MAX = 20;
    // Min and max of rectangle side lengths.
    this.RECTANGLE_SIDE_MIN = 40;
    this.RECTANGLE_SIDE_MAX = 100;
    // Min and max of triangle side lengths.
    this.TRIANGLE_SIDE_MIN = 50;
    this.TRIANGLE_SIDE_MAX = 100;

    // Canvas dimensions.
    this.w = this.canvas.width;
    this.h = this.canvas.height;
  }

  drawCircle(ctx, centerX, centerY, radius) {
    centerX = centerX == null ? this.w * Math.random() : centerX;
    centerY = centerY == null ? this.h * Math.random() : centerY;
    radius =
      radius == null
        ? this.CIRCLE_RADIUS_MIN +
          (this.CIRCLE_RADIUS_MAX - this.CIRCLE_RADIUS_MIN) * Math.random()
        : radius;

    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();
    return [0, centerX, centerY, 0];
  }

  drawLineSegment(ctx, x0, y0, x1, y1) {
    x0 = x0 == null ? Math.random() * this.w : x0;
    y0 = y0 == null ? Math.random() * this.h : y0;
    x1 = x1 == null ? Math.random() * this.w : x1;
    y1 = y1 == null ? Math.random() * this.h : y1;

    ctx.strokeStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    return [x0, x1, y0, y1];
  }

  drawRectangle(ctx, centerX, centerY, w, h) {
    w =
      Math.random() * (this.RECTANGLE_SIDE_MAX - this.RECTANGLE_SIDE_MIN) +
      this.RECTANGLE_SIDE_MIN;
    h =
      Math.random() * (this.RECTANGLE_SIDE_MAX - this.RECTANGLE_SIDE_MIN) +
      this.RECTANGLE_SIDE_MIN;
    centerX = (this.w - w) * Math.random() + w / 2;
    centerY = (this.h - h) * Math.random() + h / 2;

    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.moveTo(centerX - w / 2, centerY - h / 2);
    ctx.lineTo(centerX + w / 2, centerY - h / 2);
    ctx.lineTo(centerX + w / 2, centerY + h / 2);
    ctx.lineTo(centerX - w / 2, centerY + h / 2);
    ctx.fill();
    return [centerX - w / 2, centerX + w / 2, centerY - h / 2, centerY + h / 2];
  }

  drawTriangle(ctx, centerX, centerY, side, angle) {
    side =
      this.TRIANGLE_SIDE_MIN +
      (this.TRIANGLE_SIDE_MAX - this.TRIANGLE_SIDE_MIN) * Math.random();
    centerX = (this.w - side) * Math.random() + side / 2;
    centerY = (this.h - side) * Math.random() + side / 2;
    angle = (Math.PI / 3) * 2 * Math.random(); // 0 - 120 degrees.

    const ctrToVertex = side / 2 / Math.cos((30 / 180) * Math.PI);
    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();

    const alpha1 = angle + Math.PI / 2;
    const x1 = centerX + Math.cos(alpha1) * ctrToVertex;
    const y1 = centerY + Math.sin(alpha1) * ctrToVertex;
    const alpha2 = alpha1 + (Math.PI / 3) * 2;
    const x2 = centerX + Math.cos(alpha2) * ctrToVertex;
    const y2 = centerY + Math.sin(alpha2) * ctrToVertex;
    const alpha3 = alpha2 + (Math.PI / 3) * 2;
    const x3 = centerX + Math.cos(alpha3) * ctrToVertex;
    const y3 = centerY + Math.sin(alpha3) * ctrToVertex;

    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.fill();
    const xs = [x1, x2, x3];
    const ys = [y1, y2, y3];
    return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  }

  async generateExample() {
    const ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, this.w, this.h);

    for (let i = 0; i < getRandomInt(2, 10); ++i) {
      this.drawCircle(ctx);
      this.drawLineSegment(ctx);
    }

    return { fullCanvas: this.canvas, boundingBox: this.drawRectangle(ctx) };
  }

  drawBoundingBoxes(trueBoundingBox, predictBoundingBox, percentage) {
    tf.util.assert(
      trueBoundingBox != null && trueBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
        `but got ${trueBoundingBox.length} instead`
    );
    tf.util.assert(
      predictBoundingBox != null && predictBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
        `but got ${trueBoundingBox.length} instead`
    );

    let left = trueBoundingBox[0];
    let right = trueBoundingBox[1];
    let top = trueBoundingBox[2];
    let bottom = trueBoundingBox[3];

    const ctx = this.canvas.getContext("2d");
    ctx.beginPath();
    ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE;
    ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH;
    ctx.moveTo(left, top);
    ctx.lineTo(right, top);
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.lineTo(left, top);
    ctx.stroke();

    ctx.font = "15px Arial";
    ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
    ctx.fillText("true", left, top);

    left = predictBoundingBox[0];
    right = predictBoundingBox[1];
    top = predictBoundingBox[2];
    bottom = predictBoundingBox[3];

    ctx.beginPath();
    ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
    ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
    ctx.moveTo(left, top);
    ctx.lineTo(right, top);
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.lineTo(left, top);
    ctx.stroke();

    ctx.font = "15px Arial";
    ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
    ctx.fillText(`predicted: ${percentage.toFixed(1)}%`, left, bottom);
  }
}

module.exports = { ObjectDetectionImageSynthesizer };
