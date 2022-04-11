import sys
import numpy as np
import pandas as pd
from skimage import feature
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
from skimage.measure import label, regionprops,regionprops_table,inertia_tensor_eigvals,inertia_tensor,moments_central,moments_normalized,moments_hu,perimeter

class detection:
    def __init__(self, path, ar_no):
        self.path = path
        self.ar_no = ar_no
        
        sys.path.append('../code')
        from ts_processing import ts_processing
        from region_detection import pos_neg_detection
        self.dt = pos_neg_detection()
        self.ts = ts_processing(self.path, self.ar_no)

    def PIL_detect(self):
        self.TS_list_n, self.ls_files_n = self.ts.process()
        self.TS_list, self.PIL_series_orig_n = self.PIL_series(self.TS_list_n, self.ls_files_n, pos_g_val=100, neg_g_val=-100)
        self.ls_pil_orig_n, self.ls_label_orig_n = self.PIL_dataframe(self.PIL_series_orig_n)
        return self.ls_pil_orig_n, self.ls_label_orig_n

    def PIL_series(self, TS_list, ls_map, pos_g_val=100, neg_g_val=-100):
        new_TS_list = []
        pil_series_not_thin = []
        success_count = 0

        for i, sub_map in enumerate(ls_map):
            pos_map, neg_map = self.dt.identify_pos_neg_region(sub_map, pos_gauss=pos_g_val, neg_gauss=neg_g_val)

            pos_edge = self.dt.edge_detection(pos_map)
            neg_edge = self.dt.edge_detection(neg_map)

            pos_dil_edge = self.dt.buff_edge(pos_edge, size=4)
            neg_dil_edge = self.dt.buff_edge(neg_edge, size=4)

            pil_final_not_thin = self.PIL_extraction(pos_dil_edge, neg_dil_edge, sub_map, thinning=False)
            pil_series_not_thin.append(pil_final_not_thin)

            new_TS_list += [TS_list[i]]
            success_count += 1

        print("Success: ", success_count)
        print("Fail: ", len(ls_map) - success_count)

        return new_TS_list, pil_series_not_thin  # return the PIL series (not thinning) and submap of single HARP number

    def PIL_extraction(self, buff_pos, buff_neg, fits_map, thinning=True):
        pil_mask = np.invert(np.isnan(fits_map.data))

        pil_result = np.where(buff_pos & buff_neg & pil_mask)  # index(pixel) coordinates of PIL intersection

        pil_map = np.zeros(fits_map.data.shape)

        pil_map[pil_result[0], pil_result[1]] = 1

        if thinning == True:
            thinned_pil = thin(pil_map)
            return thinned_pil

        else:
            return pil_map

    def PIL_dataframe(self, lst_pil):
        # input: list of original PIL
        # output: list of pil dataframe
        #        list of pil label
        ls_pil_df = []
        ls_ob_labels = []

        for i, file in enumerate(lst_pil):
            if (np.all(file == False)):

                ls_pil_df.append(None)
                ls_ob_labels.append(None)

            else:
                ob_labels = label(file, connectivity=2)

                prop = ['label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'bbox', 'area',
                        'coords', 'image', 'bbox_area']

                props_table = regionprops_table(ob_labels, properties=prop)

                ls_pil_df.append(pd.DataFrame(props_table))
                ls_ob_labels.append(ob_labels)

        return ls_pil_df, ls_ob_labels  # return the list of original pil dataframe and pil label list

    def filtering_by_strength(self, threshold=0.95):
        ls_n_filtering_label = []

        for i, file in enumerate(self.ls_pil_orig_n):
            if file is not None:
                file['strength'] = self.single_strength(file, self.ls_files_n[i])
                file.sort_values(by=['strength'], ascending=False, inplace=True)

                file['cum_percent'] = file['strength'].cumsum() / sum(
                    abs(file['strength']))  # generate ['cum_percent'] column
                file['str_percent'] = file['strength'] / sum(abs(file['strength']))
                file['cut_threshold'] = file['cum_percent'] - file[
                    'str_percent']  # cut threshold (handle if exist minority PIL)
                file['strength_keep'] = file.apply(lambda row: row.cut_threshold <= threshold, axis=1)

                n_label = self.filter_strength_pil(file, threshold, self.ls_label_orig_n[
                    i])  # return the label matrix satisfy the filtering threshold

                ls_n_filtering_label.append(n_label)
            else:
                ls_n_filtering_label.append(None)

        return ls_n_filtering_label  # return filtering(by strength) label matrix

    def single_strength(self, pil_df_n, submap_series_n):
        # calculate the total Gauss of each PIL: currently use: sum(abs(pil[row,column].data))
        pil_strgt_lst = []

        for i, pil_row in pil_df_n.iterrows():
            r_idx = pil_row.coords[:, 0]
            col_idx = pil_row.coords[:, 1]

            pil_strength = sum(abs(submap_series_n.data[r_idx, col_idx]))  # single strength

            pil_strgt_lst.append(pil_strength)

        return pil_strgt_lst


    def filter_strength_pil(self, pil_df, s_t, pil_label):
        # single dataframe
        cut_label = pil_df[pil_df['cut_threshold'] > s_t].label.values  # threshold which contains 90% ~95% PIL flux

        cut_idx = np.isin(pil_label, cut_label)  # idividual pil which total strength not initial total flux

        pil_label[cut_idx] = 0  # set small PIL flux label to 0, only keep big PIL flux label

        return pil_label

    def thining_strength_label(self, ls_filter_label):
        ls_strength_binary_image = []
        ls_thin_dataframe = []
        ls_thin_binary_image = []

        for i, file in enumerate(ls_filter_label):
            if file is not None:
                ls_strength_binary_image.append(np.zeros(file.shape) + file)

                pil_thin_label = self.label_conn_thin(file)

                prop = ['label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'bbox', 'area',
                        'coords', 'image', 'bbox_area', 'perimeter', 'convex_area']
                props_table = regionprops_table(pil_thin_label, properties=prop)

                thin_binary = pil_thin_label

                ls_thin_dataframe.append(pd.DataFrame(props_table))
                ls_thin_binary_image.append(thin_binary)
            else:
                ls_strength_binary_image.append(None)
                ls_thin_dataframe.append(None)
                ls_thin_binary_image.append(None)

        return ls_strength_binary_image, ls_thin_dataframe, ls_thin_binary_image

    def label_conn_thin(self, filtered_orig_label):
        pil_thin = thin(filtered_orig_label)  # thin the original filttered label matrix, return the binary mask
        filtered_orig_label[~pil_thin] = 0  # keep the original label, and set non-thining part as 0

        return filtered_orig_label  # return the label mask after thinning after_thin_label(keep the same label id as before_thinning)

    def filtering_by_thinned_length(self, ls_thin_df, ls_thin_b_image, thin_threshold=14):
        for i, file in enumerate(ls_thin_df):
            if file is not None:
                cut_thin_label = file[file['area'] < thin_threshold].label.values  # filter pil by length threshold
                file['length_keep'] = file.apply(lambda row: row.area >= thin_threshold, axis=1)

                if len(cut_thin_label) > 0:
                    cut_idx = np.isin(ls_thin_b_image[i], cut_thin_label)
                    ls_thin_b_image[i][cut_idx] = 0  # remove thinned length after filtering

        return ls_thin_b_image  # return list of length filtering binary PIL image with label

    def get_convex_image(self, pil_binary):

        convex_b = convex_hull_image(pil_binary).astype('int')

        return convex_b

    def get_final_bib_pil(self, s_filter_b_image, final_label):

        f_lab_matrix = np.zeros(s_filter_b_image.shape) + s_filter_b_image

        remove_idx = np.isin(f_lab_matrix, final_label, invert=True)

        f_lab_matrix[remove_idx] = 0  # final binary image of PIL (after strength and thinning)

        return f_lab_matrix

    def filter_strength_length(self, thin_df):
        # filter pil based on length  label
        thin_label = set(list(thin_df[thin_df['length_keep'] == True].label.values))
        final_label = list(thin_label)

        return final_label

    def final_pil(self, ls_filter_thin_df, ls_s_filter):
        binary_PIL = []
        convex_PIL = []

        for i, file in enumerate(ls_filter_thin_df):
            if file is not None:
                f_label = self.filter_strength_length(file)
                f_lab_matrix = self.get_final_bib_pil(ls_s_filter[i], f_label)
                f_conv_image = self.get_convex_image(f_lab_matrix)
                binary_PIL.append(f_lab_matrix)
                convex_PIL.append(f_conv_image)

            else:
                binary_PIL.append(None)
                convex_PIL.append(None)

        return binary_PIL, convex_PIL

    def convex_pil_thin(self, ls_pil_thin_final):
        ls_convex_df = []
        ls_convex_thin_b = []

        for i, file in enumerate(ls_pil_thin_final):
            if file is not None:
                f_conv_image = self.get_convex_image(file)

                ob_labels = label(f_conv_image, connectivity=2)

                prop = ['label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'bbox', 'area',
                        'coords', 'image', 'bbox_area', 'perimeter']

                props_table = regionprops_table(ob_labels, properties=prop)

                ls_convex_df.append(pd.DataFrame(props_table))

                ls_convex_thin_b.append(f_conv_image)

            else:
                ls_convex_df.append(None)
                ls_convex_thin_b.append(None)

        return ls_convex_df, ls_convex_thin_b

