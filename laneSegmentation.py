import numpy as np
import cameraCalibration as cc
import lineDetection as ld
import cv2

class Line():
    def __init__(self, line_type, n_iter, yvals):
        self.n_iter = n_iter
        self.line_type = line_type
        self.yvals = yvals
        self.bestx = None
        self.best_fit = None
        self.fit = []
        self.radius = None
        self.line_pos = None

    def update_line(self, image):
        # Overall line update pipeline
        # Find line px, fit them, get parameters
        coord = self.gen_fit_data(image)
        fit = self.fit_line(coord)
        radius, line_pos = self.curve_and_pos(coord, image.shape)
        # sanity check via fit delta
        if len(self.fit) > 1: 
            delta_fit = self.best_fit - fit
        else: delta_fit = np.zeros(3)
        if np.dot(delta_fit, delta_fit) < 9999:
            self.fit.append(fit)
            if len(self.fit) > self.n_iter: self.fit.pop(0)
            self.best_fit = np.average(self.fit, axis=0)
            self.bestx = self.polynomial(self.best_fit)
            self.radius = radius
            self.line_pos = line_pos

    def guess_center(self, image):
        # Distribution of line pixels along x-direction
        image_frac=4 # fraction of image to ignore
        offset=100   # px value of manual offset if needed
        histogram = np.sum(image[int(image.shape[0] / image_frac):, :], axis=0)
        mid_point = int((histogram.shape[0] - 2 * offset) / 2)
        if self.line_type == 'L':
            index = np.arange(offset, mid_point)
        elif self.line_type == 'R':
            index = np.arange(mid_point,histogram.shape[0] - offset)
        # Calculate position of peak in the histogram using weighted average
        center = np.average(index.astype(int), weights=histogram[index]).astype(int)
        return center

    def gen_fit_data(self, image):
        hot_points = []
        peak_width=50 # widest lane line
        nbins=10 # resolution of fit
        Y, X = np.nonzero(image)
        Xy_idx = np.arange(len(Y))
        imsize = image.shape
        bin_size = int(imsize[0] / nbins)
        # Guess approximate location of the line center
        center = self.guess_center(image)
        x_start = center - peak_width
        x_end = center + peak_width
        for nbin in range(nbins):
            index = np.arange(max(0, x_start), min(x_end, imsize[1]))
            # y-direction window start and end
            y_end = imsize[0] - nbin * bin_size
            y_start = y_end - bin_size
            # Distribution of line pixels along x contained in y bin
            histogram = np.sum(image[y_start:y_end, :], axis=0)
            # Calculate line center using weighted average
            try: center = int(np.average(index, weights=histogram[index]))
            except: pass
            # Update scanning window and select points 
            x_start = center - peak_width
            x_end = center + peak_width
            idx = Xy_idx[
                (X >= x_start) & (X < x_end) & \
                (Y >= y_start) & (Y < y_end)]
            hot_points.append(idx)
        # Concatenate hot points found in y-direction scan
        hot_points = np.concatenate(hot_points)
        return X[hot_points], Y[hot_points]

    def curve_and_pos(self, position, imsize):
        ym_per_pix = 3 / 70  # m/px y dimension - sketchy estimate
        xm_per_pix = 7 / 1400  # m/px x dimension - seems legit
        yval, xval = imsize
        X, Y = position
        # Convert data from px to m
        fit_cr = np.polyfit(Y * ym_per_pix, X * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * yval * ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])
        abs_line_pos = fit_cr[0] * (yval * ym_per_pix)**2 + fit_cr[1] * yval * ym_per_pix + fit_cr[2]
        line_pos = abs_line_pos - xval * xm_per_pix / 2
        return curverad, line_pos

    def fit_line(self, position):
        X, Y = position
        return np.polyfit(Y, X, 2)

    def polynomial(self, fit):
        return fit[0] * self.yvals**2 + fit[1] * self.yvals + fit[2]


class Lane(Line):
    """
    Uses two line objects (L + R) and provides methods for lane approximation
    """
    def __init__(self, mtx, dist, n_iter=10):
        self.yvals = np.linspace(0, 720, 20)
        self.ll = Line('L', n_iter, self.yvals)
        self.rl = Line('R', n_iter, self.yvals)
        self.mtx = mtx
        self.dist = dist

    def update_lane(self, image, debug=False):
        undist, warped = self.process_image(image)
        self.ll.update_line(warped)
        self.rl.update_line(warped)
        inv_M = cc.get_transform_matrix(inverse=True)
        result = self.draw_lane(undist, warped, inv_M)
        self.write_curvature_and_position(result)
        return result

    def process_image(self, image):
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        imgline = ld.detect_line(undist)
        warped = cc.warp_image(imgline)
        warped[warped > 0] = 1
        return undist, warped

    def draw_lane(self, undist, warped, inv_M):
        left_fitx = self.ll.bestx
        right_fitx = self.rl.bestx
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Cast x and y into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.yvals])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        # reverse warp back to OG perspective
        newwarp = cv2.warpPerspective(color_warp, inv_M, (undist.shape[1], undist.shape[0]))
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def write_curvature_and_position(self, image):
        # Average radius of the left and right lane
        av_radius = (self.ll.radius + self.rl.radius) / 2.
        # Car position wrt lane center
        dist_center = (self.ll.line_pos + self.rl.line_pos) / 2.
        radius_str = 'Road radius: %d m' % av_radius
        if dist_center >=0:
            dist_center_str = '%.2f m left of center' % dist_center
        elif dist_center <0:
            dist_center_str = '%.2f m right of center' % dist_center
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(image, radius_str, (50, 75), font, 1.5, (255, 255, 255), 2)
        cv2.putText(image, dist_center_str, (50, 150), font, 1.5, (255, 255, 255), 2)